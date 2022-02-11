from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from scipy.sparse import dok_matrix
from itertools import chain
import random
import math
import torch_scatter
import time

# Deep learning model
class many2many(nn.Module):
    '''
        init_row, init_col: Size of the first seed matrix
        input_size_sec: # entries in the second seed matrix
        hidden_size: # hidden units in deep models
        k: # of products of the first embed + seed matrix
        second_k: # of products of the second embed
        g_sq_sum: the square sum of a target matrix
        fixed: true when not using parameters for g_sq_sum 
        model_type: double or float
        use_sec: true when using two matrix model.                
    '''
    def __init__(self, init_size, hidden_size, ks, g_sq_sum, fixed, model_type):
        super(many2many, self).__init__()
        self.init_size = init_size
        self.hidden_size = hidden_size
        self.order, self.ks = len(ks), [1] + ks
        
        self.input_emb = nn.Embedding(num_embeddings = self.init_size * (((self.init_size ** self.order) - 1) // (self.init_size - 1)), embedding_dim = hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, init_size ** (self.order - i)), torch.nn.Softplus()) for i in range(self.order)])
        
        self.sos = nn.Parameter(4 * torch.rand(init_size ** self.order) - 2)        
        self.sos_softplus = torch.nn.Softplus()
        
        self.fixed = fixed
        level_sq_sum = math.exp(math.log(g_sq_sum) / self.ks[-1])              
        if fixed: self.level_sq_sum = level_sq_sum
        else: 
            if model_type == "double": self.level_sq_sum = nn.Parameter(torch.DoubleTensor([level_sq_sum])) 
            elif model_type == "float": self.level_sq_sum = nn.Parameter(torch.FloatTensor([level_sq_sum])) 
     
    '''
        _input: batch size x seq_len
        return value: Probability of input
    '''
    def forward(self, _input):
        # Run model
        _input = _input.transpose(0, 1)
        _, batch_size = _input.size()
        rnn_output, _ = self.rnn(self.input_emb(_input[:-1]))        # seq_len - 1 x batch_size x hidden_size
        
        # Fix average and scale of sos 
        sos = self.sos_softplus(self.sos)
        if self.fixed: sos_scaled = (sos / torch.sqrt(torch.sum(torch.square(sos)))) * math.sqrt(self.level_sq_sum)
        else: sos_scaled = (sos / torch.sqrt(torch.sum(torch.square(sos)))) * torch.sqrt(self.level_sq_sum)
        total_output = sos_scaled[_input[0, :]]
        
        start_idx = 0
        for i in range(self.order):
            if self.ks[i] < self.ks[i+1]:
                _output = self.linears[i](rnn_output[self.ks[i]-1:self.ks[i+1]-1]).view(self.ks[i+1] - self.ks[i], batch_size, -1)   # k[i+1} - k[i] x batch_size x 2**(order - i)
                sq_sum_root = torch.sqrt(torch.sum(torch.square(_output), dim=-1, keepdim=True))
                _output = _output / sq_sum_root
                _tidxs = (_input[self.ks[i]:self.ks[i+1]].unsqueeze(-1) - start_idx)   # k[i+1]-k[i] x batch_size x 1
                _output_end = torch.gather(_output, dim=-1, index=_tidxs).squeeze(-1)  # k[i+1]-k[i] x batch_size
                total_output = total_output * torch.prod(_output_end, 0) * (self.level_sq_sum ** ((self.ks[i+1] - self.ks[i])/2.))
            start_idx += (self.init_size ** (self.order - i))
        return total_output
    
    
class k_mat(nn.Module):
    '''
        init_row, init_col: Size of the first seed matrix        
        k: # of products of the first embed + seed matrix
        second_k: # of products of the second embed
        g_sq_sum: the square sum of a target matrix   
        fixed: true when not using parameters for g_sq_sum  
        use_sec: true when using two matrix model.                
    '''
    def __init__(self, init_size, ks, g_sq_sum, fixed, use_sec=True):
        # set Seed matrix
        super(k_mat, self).__init__()
        self.order, self.ks = len(ks), [0] + ks
        self.init_size = init_size
        self.mats = nn.ParameterList([nn.Parameter(torch.rand(init_size ** (self.order - i))) for i in range(self.order)])
        level_sq_sum = math.exp(math.log(g_sq_sum) / self.ks[-1])
        if fixed: self.level_sq_sum = level_sq_sum
        else: self.level_sq_sum = nn.Parameter(torch.FloatTensor([level_sq_sum])) 
            
    '''
        _input: batch size x seq_len
        return value: Probability of input
    '''            
    def forward(self, _input):
        total_output = torch.ones(_input.shape[0]).to(_input.device)
        start_idx = 0
        # print(_input)
        for i in range(self.order):
            if self.ks[i] < self.ks[i+1]:                 
                #print(f'k start: {self.ks[i]-1}, k end: {self.ks[i+1]-1}')
                #print(f'start idx: {start_idx}')
                #print(_input[:, self.ks[i]-1:self.ks[i+1]-1] - start_idx)     
                _output = torch.prod(self.mats[i][_input[:, self.ks[i]:self.ks[i+1]] - start_idx], dim=1)
                _output = _output / (torch.sum(torch.square(self.mats[i]))**((self.ks[i+1] - self.ks[i])/2.))
                _output = _output * (self.level_sq_sum ** ((self.ks[i+1] - self.ks[i])/2.))
                total_output = total_output * _output
            start_idx += (self.init_size ** (self.order - i))
        return total_output
    
# Total model that includes permutation
class kronecker_model:
    '''
        graph: input graph
        init_row: number of row in the first seed matrix
        init_col: number of col in all seed matrices.
        k: number of products of the first matirx
        second_k: number of productes of teh second matrix
        device: torch device that runs gpu                
        sample_weight: weight for designing threshold function
        fixed: use fixed sqaure sum        
    '''
    def __init__(self, graph, init_size, ks, device, sample_weight, fixed):
        self.init_size = init_size
        self.order = len(ks)
        self.ks = ks
        self.device, self.graph = device, graph    
        # Initialize device
        use_cuda = torch.cuda.is_available()
        self.i_device = torch.device("cuda:" + str(self.device[0]) if use_cuda else "cpu")  
        
        self.bases = [(self.init_size ** ((_k - 1) - torch.arange(ks[-1]))).to(self.i_device) for _k in ks]
        # self.bases = (self.init_size ** ((ks[-1] - 1) - torch.arange(ks[-1]))).to(self.i_device)
        self.dims = (2 ** np.array(ks)).tolist()
        with torch.no_grad():
            self.pads = torch.zeros(ks[-1], dtype=torch.long).to(self.i_device) 
            for i in range(self.order):
                self.bases[i][ks[i]:ks[-1]] = self.dims[-1]
                self.pads[ks[i]:ks[-1]] += self.init_size ** (self.order - i)
            #print(self.bases, self.dims, self.pads)
        
        self.sample_weight, self.fixed = sample_weight, fixed
        self.stat_mu = np.mean(self.graph.indices[-1])
        self.stat_var = np.var(self.graph.indices[-1], dtype=np.float64) * self.graph.real_num_nonzero
        self.indices = [self.graph.indices[:, i] for i in range(self.order)]
    
    # Initialize the deep learning model
    def init_model(self, hidden_size, model_type, data_type):
        if model_type == "many2many":
            self.model = many2many(self.init_size, hidden_size, self.ks, self.graph.sq_sum, self.fixed, data_type)
        elif model_type == "two_mat":
            self.model = k_mat(self.init_size, self.ks, self.graph.sq_sum, self.fixed)
        else:
            assert(False)
        
        self.data_type = data_type        
        if data_type == "double": self.model.double()
        elif data_type == "float": self.model.float()
        if len(self.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids = self.device)                        
        self.model = self.model.to(self.i_device)
        print(f"The number of params:{ sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    '''
        Load the permutation
    '''
    def init_permutation(self):        
        self.perms = [torch.randperm(_dim).to(self.i_device) for _dim in self.dims]

    '''
        Set the permtuation (e.g. from node ID to row index) from file
        row_perm_file, col_perm_file: file that saves initalized permutation
    '''
    def set_permutation(self, row_perm_file, col_perm_file):                      
        # Read permutations from files
        with open(row_perm_file, 'r') as ff:            
            init_row_perm = [(int(val)-1) for val in ff]            
        with open(col_perm_file, 'r') as ff:
            init_col_perm = [(int(val)-1) for val in ff]        
                
        print(f'row min: {min(init_row_perm)}, \
              row max: {max(init_row_perm)}, row avg:{sum(init_row_perm) / len(init_row_perm)}')
        print(f'col min: {min(init_col_perm)}, \
              col max: {max(init_col_perm)}, row avg:{sum(init_col_perm) / len(init_col_perm)}')
        
        # Set the node permutatoin (node -> row)
        while len(init_row_perm) < self.num_row: init_row_perm.append(len(init_row_perm))
        while len(init_col_perm) < self.num_col: init_col_perm.append(len(init_col_perm))    
        self.perms = [torch.LongTensor(init_row_perm).to(self.i_device), torch.LongTensor(init_col_perm).to(self.i_device)]      
    
    def predict_batch(self, batched_indices):
        curr_batch_size = batched_indices.shape[0]
        with torch.no_grad():
            inputs = torch.zeros(curr_batch_size, self.ks[-1], dtype=torch.long).to(self.i_device)   # batch size x k
            for i in range(self.order):
                _idxs = torch.LongTensor(batched_indices[:, i]).to(self.i_device)   
                _idxs = (self.perms[i][_idxs].unsqueeze(1) // self.bases[i]) % self.init_size      # batch size x k
                inputs = inputs * self.init_size + _idxs   # 
                # inputs = inputs * self.init_size + _idxs
            # print(inputs.shape, inputs.max(), inputs.min())
            inputs = inputs + self.pads.unsqueeze(0)
            # print(inputs.shape, inputs.max(dim=0), inputs.min(dim=0))
        return self.model(inputs)
    
    '''
        Compute the L2 loss in an efficient manner
        is_train: set True if it is in the training process
        batch_size: batch size for nonzeros        
    '''
    def L2_loss(self, is_train, batch_size):        
        # nnzs = list(zip(graph.adjcoo.row, graph.adjcoo.col))
        if len(self.device) > 1: loss = self.model.module.level_sq_sum ** self.ks[-1]
        else: loss = self.model.level_sq_sum ** self.ks[-1]
        if not self.fixed:              
            if is_train:
                loss.backward()
            loss = loss.item()

        for i in range(0, self.graph.real_num_nonzero, batch_size):    
            # Extract nodes and edges
            curr_batch_size = min(batch_size, self.graph.real_num_nonzero - i)
            preds = self.predict_batch(self.graph.indices[i:i+curr_batch_size]).double()
            vals = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device).double()            
            curr_loss = (torch.square(preds - vals) - torch.square(preds)).sum()                           
            loss += curr_loss.item()       
            if is_train:
                curr_loss.backward()
        return loss
    
    '''
        Compute the L2 loss in a naive manner
        batch_size: batch size for nonzeros       
    '''
    def loss_naive(self, batch_size):
        loss_sum = 0.0   
        dense_gt = np.zeros(self.dims)
        # print(dense_gt.shape)
        for i in range(self.graph.real_num_nonzero):
            dense_gt[self.graph.indices[i][0], self.graph.indices[i][1], self.graph.indices[i][2]] = self.graph.val[i]
        dense_gt = dense_gt.reshape(self.dims[0], -1)
        # print(dense_gt.shape)
        
        if len(self.device) > 1: self.model.module.eval()
        else: self.model.eval()
        with torch.no_grad():
            # Handle zero early, build index for row
            # Assume entries are all zero, compute loss
            for i in tqdm(range(self.dims[0])):
                batched_indices = np.array([[i, _ // self.dims[2], _ % self.dims[2]] for _ in range(self.dims[1] * self.dims[2])])
                outputs = self.predict_batch(batched_indices).double()
                curr_loss = torch.square(outputs - torch.DoubleTensor(dense_gt[i]).to(outputs.device)).sum()
                loss_sum += curr_loss.item()
        return loss_sum 
    
        '''
        Sample node permutation
        batch_size: batch size for nonzeros        
    '''
    def update_perm(self, coor, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: node idx -> pair idx                
            mapping: node idx -> node idx
        '''
        start_time = time.time()
        sampled_n = self.dims[coor]       
        samples = torch.randperm(sampled_n).to(self.i_device)
        
        start_time = time.time()
        pair_idx = torch.arange(sampled_n).to(self.i_device)
        pair_idx[samples] = torch.arange(sampled_n).to(self.i_device)
        pair_idx //= 2
        mapping = torch.arange(sampled_n).to(self.i_device)
        mapping[samples] = torch.stack((samples[1::2], samples[0::2]), dim=1).view(-1) 
        #print(f'build mapping: {time.time() - start_time}')        
    
        '''
            sampled_points: pair idx -> sampled p
            prob_before, prob_after: pair idx -> prob sum of the pair of nodes before, after
        '''
        start_time = time.time()
        if self.data_type == "double": curr_dtype = torch.double                       
        elif self.data_type == "float": curr_dtype = torch.float        
        else: assert(False)
        sampled_points = torch.rand(sampled_n // 2, dtype=curr_dtype).to(self.i_device)
        prob_ratios = torch.zeros((sampled_n // 2), dtype=torch.double).to(self.i_device)
        # Compute the change of loss                                
        # non zero part (No need to consider zero part because prob_before and prob_after are the same when all entries are zero)
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            # Build an input for the current batch
            curr_batch_size = min(batch_size, num_nnz - i)
            coor_indices = self.indices[coor][i:i+curr_batch_size]
            vals = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device).double()
            curr_pair = pair_idx[coor_indices]
            
            batched_indices = self.graph.indices[i:i+curr_batch_size]
            preds_before = self.predict_batch(batched_indices).double()
            batched_indices_new = np.copy(batched_indices)
            batched_indices_new[:, coor] = mapping.detach().cpu().numpy()[coor_indices]
            preds_after = self.predict_batch(batched_indices_new).double()
            prob_ratios.index_add_(0, curr_pair, (torch.square(preds_after)-torch.square(preds_before)) + (torch.square(preds_before - vals)-torch.square(preds_after - vals)))                           
        
        prob_ratios.clamp_(min=-(10**15))
        prob_thre = prob_ratios.clone()
        final_decision = (prob_thre >= 0.) | (sampled_points <= (self.sample_weight * prob_thre).exp())
        #print(f'node compute loss: {time.time() - start_time}')
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        self.perms[coor][ll], self.perms[coor][rr] = self.perms[coor][rr].clone(), self.perms[coor][ll].clone()     
        #print(f'node change index: {time.time() - start_time}\n')
        return prob_ratios[final_decision].sum().item()
    
# Model that uses simple algorithm for finding the best permutation
class perm_model(kronecker_model):
    def __init__(self, graph, init_size, ks, device, sample_weight, fixed):
        super().__init__(graph, init_size, ks, device, sample_weight, fixed)            
        self.hashs = [None for i in range(self.order)]        
        
    def update_hashs(self):
        for hash_fn in self.hashs:
            del hash_fn
        self.hashs = [None for i in range(self.order)]
        for i in range(self.order):
            self.hashs[i] = torch.randperm(self.dims[i]).to(self.i_device) + 1
        
    '''
        Sample node permutation
        batch_size: batch size for nonzeros        
    '''
    def update_perm(self, coor, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: node idx -> pair idx                
            mapping: node idx -> node idx
        '''
        start_time = time.time()
        sampled_n = self.dims[coor]
        target_digit = np.random.randint(sampled_n)
        target_digit = target_digit & (-target_digit)
        if target_digit == 0: target_digit = sampled_n // 2
            
        sampled_kro_coors = torch.randperm(sampled_n // 2).to(self.i_device)
        sampled_kro_coors = (sampled_kro_coors * 2) - (sampled_kro_coors & (target_digit - 1)) + (target_digit * (torch.rand(sampled_n // 2).to(self.i_device) < 0.5).long())
        perm_inv = torch.empty_like(self.perms[coor])
        perm_inv[self.perms[coor]] = torch.arange(sampled_n).to(self.i_device)
        
        sampled_coors = perm_inv[sampled_kro_coors]  # idx -> row
        sampled_coors_mate = perm_inv[sampled_kro_coors ^ target_digit]      
        #print(f'min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        maxh = torch.zeros_like(sampled_coors)
        for i in range(self.order):
            if i == coor: continue
            _partial_maxh = torch_scatter.scatter(self.hashs[i][self.indices[i]], torch.LongTensor(self.indices[coor]).to(self.i_device), dim=-1, dim_size=self.dims[coor], reduce='max')[sampled_coors]         
            maxh = (maxh * self.dims[i]) + _partial_maxh
            
        #print(f'node min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        maxh = (maxh * self.dims[coor]) + torch.arange(sampled_n // 2).to(self.i_device)
        sorted_vals, sorted_idx = torch.sort(maxh)
        sorted_vals = sorted_vals.detach().cpu().numpy()
        samples = np.zeros(sampled_n, dtype=np.int64)
        _ss, _ee, _idx = 0, (sampled_n // 2) - 1, 0
        while _idx < (sampled_n // 2):
            if ((_idx + 1) >= (sampled_n // 2)) or ((sorted_vals[_idx] // self.dims[coor]) != (sorted_vals[_idx + 1] // self.dims[coor])):
                samples[_ee * 2] = sampled_coors[sorted_idx[_idx]]
                samples[_ee * 2 + 1] = sampled_coors_mate[sorted_idx[_idx]]
                _idx, _ee = _idx + 1, _ee - 1
            else:
                samples[_ss * 2] = sampled_coors[sorted_idx[_idx]]
                samples[_ss * 2 + 1] = sampled_coors_mate[sorted_idx[_idx + 1]]
                samples[_ss * 2 + 2] = sampled_coors[sorted_idx[_idx + 1]]
                samples[_ss * 2 + 3] = sampled_coors_mate[sorted_idx[_idx]]
                _idx, _ss = _idx + 2, _ss + 2
        
        samples = torch.LongTensor(samples).to(self.i_device)
        shuffle_idx = torch.cat((torch.arange(_ss * 2).to(self.i_device), (_ss * 2) + torch.randperm(sampled_n - (_ss * 2)).to(self.i_device)))
        samples = samples[shuffle_idx]
        # _, sorted_idx = torch.sort(maxh, stable=True)
        # samples = torch.zeros(sampled_n, dtype=torch.long).to(self.i_device)
        # samples[0::4], samples[1::4] = sampled_coors[sorted_idx[0::2]], sampled_coors_mate[sorted_idx[1::2]]
        # samples[2::4], samples[3::4] = sampled_coors[sorted_idx[1::2]], sampled_coors_mate[sorted_idx[0::2]]
        #print(f'node handle remaining: {time.time() - start_time}, paired: {_ss * 2}') 
        
        start_time = time.time()
        pair_idx = torch.arange(sampled_n).to(self.i_device)
        pair_idx[samples] = torch.arange(sampled_n).to(self.i_device)
        pair_idx //= 2
        mapping = torch.arange(sampled_n).to(self.i_device)
        mapping[samples] = torch.stack((samples[1::2], samples[0::2]), dim=1).view(-1) 
        #print(f'build mapping: {time.time() - start_time}')        
    
        '''
            sampled_points: pair idx -> sampled p
            prob_before, prob_after: pair idx -> prob sum of the pair of nodes before, after
        '''
        start_time = time.time()
        if self.data_type == "double": curr_dtype = torch.double                       
        elif self.data_type == "float": curr_dtype = torch.float        
        else: assert(False)
        sampled_points = torch.rand(sampled_n // 2, dtype=curr_dtype).to(self.i_device)
        prob_ratios = torch.zeros((sampled_n // 2), dtype=torch.double).to(self.i_device)
        # Compute the change of loss                                
        # non zero part (No need to consider zero part because prob_before and prob_after are the same when all entries are zero)
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            # Build an input for the current batch
            curr_batch_size = min(batch_size, num_nnz - i)
            coor_indices = self.indices[coor][i:i+curr_batch_size]
            vals = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device).double()
            curr_pair = pair_idx[coor_indices]
            
            batched_indices = self.graph.indices[i:i+curr_batch_size]
            preds_before = self.predict_batch(batched_indices).double()
            batched_indices_new = np.copy(batched_indices)
            batched_indices_new[:, coor] = mapping.detach().cpu().numpy()[coor_indices]
            preds_after = self.predict_batch(batched_indices_new).double()
            prob_ratios.index_add_(0, curr_pair, (torch.square(preds_after)-torch.square(preds_before)) + (torch.square(preds_before - vals)-torch.square(preds_after - vals)))                           
        
        prob_ratios.clamp_(min=-(10**15))
        prob_thre = prob_ratios.clone()
        final_decision = (prob_thre >= 0.) | (sampled_points <= (self.sample_weight * prob_thre).exp())
        #print(f'node compute loss: {time.time() - start_time}')
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        self.perms[coor][ll], self.perms[coor][rr] = self.perms[coor][rr].clone(), self.perms[coor][ll].clone()     
        #print(f'node change index: {time.time() - start_time}\n')
        return prob_ratios[final_decision].sum().item()