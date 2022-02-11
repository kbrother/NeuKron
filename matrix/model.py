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
    def __init__(self, init_row, init_col, input_size_sec, hidden_size, k, second_k, g_sq_sum, fixed, model_type, use_sec=True):
        super(many2many, self).__init__()
        # For first layer
        self.input_size = init_row * init_col
        self.hidden_size = hidden_size
        self.input_emb = nn.Embedding(num_embeddings = self.input_size + input_size_sec, embedding_dim = hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Sequential(nn.Linear(hidden_size, self.input_size), torch.nn.Softplus())                                                
        if use_sec:
            self.linear_sec = nn.Sequential(nn.Linear(hidden_size, input_size_sec), torch.nn.Softplus())                
        self.sos = nn.Parameter(4*torch.rand(init_row * init_col) - 2)        
        self.sos_softplus = torch.nn.Softplus()                      
        self.k, self.init_row, self.init_col = k, init_row, init_col
        self.second_k, self.use_sec = second_k, use_sec               
        self.fixed = fixed
        #self.sq_sum = nn.Parameter(torch.DoubleTensor([g_sq_sum]))
        level_sq_sum = math.exp(math.log(g_sq_sum) /(k + second_k))                      
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
        output, _ = self.rnn(self.input_emb(_input[:-1]))        # seq_len - 1 x batch_size x hidden_size
        output_front = self.linear(output[:self.k-1]).view(self.k-1, batch_size, -1)  # k - 1 x batch_size x input_size(init_row x init_col)
        sq_sum_root = torch.sqrt(torch.sum(torch.square(output_front), dim=-1, keepdim=True))
        output_front = output_front / sq_sum_root
        
        # Extract target entry
        output_front = torch.gather(output_front, dim=-1, index=_input[1:self.k].unsqueeze(-1)).squeeze(-1)    # k - 1 x batch_size     
        
        # Fix average and scale of sos 
        sos = self.sos_softplus(self.sos)
        sos_scaled = sos / torch.sqrt(torch.sum(torch.square(sos)))
        
        # Fix average and scale of end outputs
        return_val = torch.prod(output_front, 0) * sos_scaled[_input[0, :]]    # batch_size  
        if self.use_sec:
            output_end = self.linear_sec(output[self.k-1:])   # second_k x batch size x init_col 
            sq_sum_root = torch.sqrt(torch.sum(torch.square(output_end), dim=-1, keepdim=True))
            output_end = output_end / sq_sum_root
            
            # Extract target entry
            output_end = torch.gather(output_end, dim=-1, index=_input[self.k:].unsqueeze(-1) - self.input_size).squeeze(-1)
            return_val = return_val * torch.prod(output_end, 0)
        
        if self.fixed:
            multi = math.sqrt(math.exp((self.k+self.second_k) * math.log(self.level_sq_sum)))
        else:
            multi = torch.sqrt(torch.exp((self.k+ self.second_k) * torch.log(self.level_sq_sum)))
        return multi * return_val 
    

class two_mat(nn.Module):
    '''
        init_row, init_col: Size of the first seed matrix        
        k: # of products of the first embed + seed matrix
        second_k: # of products of the second embed
        g_sq_sum: the square sum of a target matrix   
        fixed: true when not using parameters for g_sq_sum  
        use_sec: true when using two matrix model.                
    '''
    def __init__(self, init_row, init_col, k, second_k, g_sq_sum, fixed, use_sec=True):
        # set Seed matrix
        super(two_mat, self).__init__()
        self.first_mat = nn.Parameter(torch.rand(init_row*init_col))        
        self.sec_huddle = init_row*init_col
        self.k, self.second_k, self.use_sec = k, second_k, use_sec
        if use_sec:
            self.second_mat = nn.Parameter(torch.rand(init_col))            
        
        # Set level sq sum
        level_sq_sum = math.exp(math.log(g_sq_sum)/(k + second_k))                
        if fixed: self.level_sq_sum = level_sq_sum
        else: 
            self.level_sq_sum = nn.Parameter(torch.FloatTensor([level_sq_sum])) 

    '''
        _input: batch size x seq_len
        return value: Probability of input
    '''            
    def forward(self, _input):        
        _output = torch.prod(self.first_mat[_input[:, :self.k]], dim=1)
        _output = _output / torch.sum(torch.square(self.first_mat))**(self.k/2.)
        if self.use_sec:
            _output = _output * torch.prod(self.second_mat[(_input[:, self.k:]-self.sec_huddle)], dim=1)
            _output = _output / torch.sum(torch.square(self.second_mat))**(self.second_k/2.)
        return _output * (self.level_sq_sum) ** ((self.k + self.second_k)/2.)
    
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
    def __init__(self, graph, init_row, init_col, k, second_k, device, sample_weight, fixed):
        self.init_row, self.init_col = init_row, init_col        
        self.k, self.second_k = k, second_k
        self.device, self.graph = device, graph    
        # Initialize device
        use_cuda = torch.cuda.is_available()
        self.i_device = torch.device("cuda:" + str(self.device[0]) if use_cuda else "cpu")  
        
        self.node_bases = init_row ** ((self.k-1) - torch.arange(self.k)).to(self.i_device)
        self.edge_bases = init_col ** ((self.k + self.second_k -1) - torch.arange(self.k + self.second_k)).to(self.i_device)
        self.num_row, self.num_col = init_row**k, init_col**(k + second_k)          
        self.sample_weight, self.fixed = sample_weight, fixed
        self.stat_mu = np.mean(self.graph.val)
        self.stat_var = np.var(self.graph.val, dtype=np.float64) * self.graph.real_num_nonzero
    
    # Initialize the deep learning model
    def init_model(self, hidden_size, model_type, data_type, use_sec):
        if model_type == "many2many":           
            self.model = many2many(self.init_row, self.init_col, self.init_col, hidden_size, self.k, self.second_k, \
                                   self.graph.sq_sum, self.fixed, data_type, use_sec)
        elif model_type == "two_mat":
            self.model = two_mat(self.init_row, self.init_col, self.k, self.second_k, self.graph.sq_sum, self.fixed, use_sec)
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
        self.node_perm = torch.randperm(self.num_row).to(self.i_device)
        self.edge_perm = torch.randperm(self.num_col).to(self.i_device)


    '''
        Set the permtuation (e.g. from node ID to row index) from file
        row_perm_file, col_perm_file: file that saves initalized permutation
    '''
    def set_permutation(self, row_perm_file, col_perm_file):                      
        # Read permutations from files
        with open(row_perm_file, 'r') as ff:            
            init_row_perm = [int(val) for val in ff]            
        with open(col_perm_file, 'r') as ff:
            init_col_perm = [int(val) for val in ff]        
                
        print(f'row min: {min(init_row_perm)}, \
              row max: {max(init_row_perm)}, row avg:{sum(init_row_perm) / len(init_row_perm)}')
        print(f'col min: {min(init_col_perm)}, \
              col max: {max(init_col_perm)}, row avg:{sum(init_col_perm) / len(init_col_perm)}')
        
        # Set the node permutatoin (node -> row)
        while len(init_row_perm) < self.num_row: init_row_perm.append(len(init_row_perm))
        while len(init_col_perm) < self.num_col: init_col_perm.append(len(init_col_perm))
            
        self.node_perm = torch.LongTensor(init_row_perm).to(self.i_device)
        self.edge_perm = torch.LongTensor(init_col_perm).to(self.i_device)        
        
        '''
        self.node_perm = torch.empty_like(node_perm)
        self.edge_perm = torch.empty_like(edge_perm)
        
        self.node_perm[node_perm] = torch.arange(self.num_row).to(self.device)
        self.edge_perm[edge_perm] = torch.arange(self.num_col).to(self.device)
        '''
        
    '''
        Compute the L2 loss in an efficient manner
        is_train: set True if it is in the training process
        batch_size: batch size for nonzeros        
    '''
    def L2_loss(self, is_train, batch_size):        
        # nnzs = list(zip(graph.adjcoo.row, graph.adjcoo.col))
        if len(self.device) > 1: loss = (self.model.module.level_sq_sum)**(self.k + self.second_k)
        else: loss = (self.model.level_sq_sum)**(self.k + self.second_k)
        if not self.fixed:              
            if is_train:
                loss.backward()
            loss = loss.item()
            
        #for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):
        for i in range(0, self.graph.real_num_nonzero, batch_size):    
            # Extract nodes and edges
            curr_batch_size = min(batch_size, self.graph.real_num_nonzero - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                          
            curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
            
            # Convert to lstm inputs
            row_idx = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
            col_idx = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
            row_idx_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
            row_idx = torch.cat((row_idx, row_idx_pad), dim=-1)

            # Correct non-zero parts
            inputs = row_idx * self.init_col + col_idx
            samples = self.model(inputs)
            #print(f'sample shape: {samples.shape}, value sahpe: {curr_val.shape}')      
            curr_loss = (torch.square(samples - curr_val) - torch.square(samples)).sum()                           
            loss += curr_loss.item()       
            if is_train:
                curr_loss.backward()

        return loss
    
    def write_matrix(self, batch_size, file_name):
        num_printed_entry = 0
        num_entry = self.graph.num_row * self.graph.num_col
        pbar = tqdm(total=num_entry)
        vals = np.zeros(num_entry)
        graph_row, graph_col, graph_val = np.array(self.graph.row_idx), np.array(self.graph.col_idx), np.array(self.graph.val)
        vals[graph_row*self.graph.num_col + graph_col] = graph_val
        
        _se = 0.
        with torch.no_grad():
            with open(file_name, 'w') as f:            
                while num_printed_entry < num_entry:            
                    if batch_size > num_entry - num_printed_entry: 
                        batch_size = num_entry - num_printed_entry

                    # Build LSTM inputs
                    curr_rows, curr_cols = torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.i_device), torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.i_device)                    
                    curr_vals = torch.tensor(vals[num_printed_entry:num_printed_entry  + batch_size]).to(self.i_device)
                    curr_rows, curr_cols = curr_rows//self.graph.num_col, curr_cols%self.graph.num_col                    
                    curr_rows, curr_cols = self.node_perm[curr_rows], self.edge_perm[curr_cols]            

                    row_idx = curr_rows.unsqueeze(1) // self.node_bases % self.init_row
                    col_idx = curr_cols.unsqueeze(1) // self.edge_bases % self.init_col            
                    row_idx_pad = self.init_row * torch.ones([batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
                    row_idx = torch.cat((row_idx, row_idx_pad), dim=-1)

                    # Get lstm outputs
                    inputs = row_idx * self.init_col + col_idx 
                    samples = self.model(inputs)
                    #print(samples.shape)
                    _se += torch.sum((samples - curr_vals)**2).item()
                    
                    samples = samples.detach().cpu().numpy()
                    for i in range(batch_size):
                        f.write(f'{samples[i]}\t')
                        if ((i+num_printed_entry+1)%self.graph.num_col == 0):
                            f.write('\n')

                    num_printed_entry += batch_size            
                    pbar.update(batch_size)
        
        print(f'loss for the parts correpsond to real matrix {_se}')

    '''
        Compute the L2 loss in a naive manner
        batch_size: batch size for nonzeros       
    '''
    def loss_naive(self, batch_size):
        loss_sum = 0.0   
        if len(self.device) > 1: self.model.module.eval()
        else: self.model.eval()
        with torch.no_grad():
            # Handle zero early, build index for row
            curr_row = self.node_perm.unsqueeze(1) // self.node_bases % self.init_row
            curr_row_pad = self.init_row * torch.ones([self.num_row, self.second_k], dtype=torch.long).to(self.i_device)    
            curr_row = torch.cat((curr_row, curr_row_pad), dim=-1)        
            # Assume entries are all zero, compute loss
            for i in tqdm(range(self.num_col)):
                curr_col = self.edge_perm[i].unsqueeze(-1) // self.edge_bases % self.init_col
                inputs = curr_row * self.init_col + curr_col                                                 
                outputs = self.model(inputs)            
                curr_loss = (torch.square(outputs)).sum()
                loss_sum += curr_loss.item()

            # Handle non-zero part            
            for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):
                # Extract nodes and edges
                curr_batch_size = min(batch_size, self.graph.real_num_nonzero - i)
                nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
                nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                            
                curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
                
                # Convert to row and col
                curr_row = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
                curr_col = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
                curr_row_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
                curr_row = torch.cat((curr_row, curr_row_pad), dim=-1)

                # Compute loss
                inputs = curr_row * self.init_col + curr_col      
                samples = self.model(inputs)
                curr_loss = (torch.square(samples - curr_val) - torch.square(samples)).sum()      
                    
                loss_sum += curr_loss.item()       
        return loss_sum 
    
    '''
        Sample node permutation
        batch_size: batch size for nonzeros        
    '''
    def sample_node_batches(self, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: node idx -> pair idx                
            mapping: node idx -> node idx
        '''
        start_time = time.time()
        samples = torch.randperm(self.num_row).to(self.i_device)
        pair_idx = torch.arange(self.num_row).to(self.i_device)
        pair_idx[samples] = torch.arange(len(samples)).to(self.i_device)
        pair_idx //= 2
        mapping = torch.arange(len(samples)).to(self.i_device)
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
        sampled_points = torch.rand(self.num_row // 2, dtype=curr_dtype).to(self.i_device)
        prob_before = torch.zeros((self.num_row // 2), dtype=curr_dtype).to(self.i_device)
        prob_after = torch.zeros((self.num_row // 2), dtype=curr_dtype).to(self.i_device)

        # Compute the change of loss                                
        # non zero part (No need to consider zero part because prob_before and prob_after are the same when all entries are zero)
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            # Build an input for the current batch
            curr_batch_size = min(batch_size, num_nnz - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                   
            cols = self.edge_perm[edges]
            col_idx = cols.unsqueeze(1) // self.edge_bases % self.init_col
            row_idx_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
            curr_pair = pair_idx[nodes]
            curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
            
            # Compute prob_before & prob_after                                   
            temp_list = [(self.node_perm[nodes], prob_before), (self.node_perm[mapping[nodes]], prob_after)]
            for (curr_rows, curr_prob) in temp_list:                                                                              
                row_idx = curr_rows.unsqueeze(1) // self.node_bases % self.init_row                                        
                row_idx = torch.cat((row_idx, row_idx_pad), dim=-1)            
                inputs = row_idx * self.init_col + col_idx         
                outputs = self.model(inputs)             
                # Why the sign is like this?
                # Because metropolis-hasting use probability, decrease of loss means the increase of prob (-loss)
                curr_prob.index_add_(0, curr_pair, torch.square(outputs) - torch.square(outputs - curr_val))                                                                    

        prob_ratios = prob_after - prob_before
        prob_ratios.clamp_(min=-(10**15))    
        prob_thre = prob_ratios.clone()
        final_decision = (prob_thre >= 0.) | (sampled_points <= (self.sample_weight * prob_thre).exp())
        #print(f'node compute loss: {time.time() - start_time}')
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        #ll, rr = map(lambda _x: np.array(_x), zip(*(samples.view(-1, 2)[final_decision].detach().cpu().numpy())))
        self.node_perm[ll], self.node_perm[rr] = self.node_perm[rr].clone(), self.node_perm[ll].clone()     
        #print(f'node change index: {time.time() - start_time}\n')
        return prob_ratios[final_decision].sum().item()
    
    '''
        Sample edge permutation
        batch_size: batch size for nonzeros       
    '''
    def sample_edge_batches(self, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: edge idx -> pair idx
            mapping: edge idx -> edge idx
        '''
        start_time = time.time()
        samples = torch.randperm(self.num_col).to(self.i_device)      
       
        pair_idx = samples.clone()
        pair_idx[samples] = torch.arange(len(samples)).to(self.i_device)
        pair_idx //= 2
        mapping = torch.arange(len(samples)).to(self.i_device)
        mapping[samples] = torch.stack((samples[1::2], samples[0::2]), dim=1).view(-1)
        #print(f'edge build mapping: {time.time() - start_time}')     

        '''
            sampled_points: pair idx -> sampled p
            prob_ratios: pair idx -> prob changes of the pair
        '''
        start_time = time.time()
        if self.data_type == "double": curr_dtype = torch.double                       
        elif self.data_type == "float": curr_dtype = torch.float        
        else: assert(False)        
        sampled_points = torch.rand(self.num_col // 2, dtype=curr_dtype).to(self.i_device)
        prob_ratios = torch.zeros((self.num_col // 2), dtype=curr_dtype).to(self.i_device)
        lb = -(10**9)

        # Compute the change of loss                                
        # non zero part
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            curr_batch_size = min(batch_size, num_nnz - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.device[0]), torch.LongTensor(edges).to(self.device[0])               
            curr_row = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
            curr_col = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
            curr_row_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.device[0])        
            curr_row = torch.cat((curr_row, curr_row_pad), dim=-1)
            curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.device[0])            
            
            neg_edges = mapping[edges]
            neg_col = self.edge_perm[neg_edges].unsqueeze(1) // self.edge_bases % self.init_col            

            inputs = torch.cat(((curr_row * self.init_col + curr_col), (curr_row * self.init_col + neg_col)), dim=0)                 
            outputs = self.model(inputs).view(2, -1)                      
            prob_ratios.index_add_(0, pair_idx[edges], (-torch.square(outputs[0]) + torch.square(outputs[0] - curr_val) - torch.square(outputs[1] - curr_val) + torch.square(outputs[1])))

        prob_ratios.clamp_(min=-(10**15))
        final_decision = (prob_ratios >= 0.) | (sampled_points <= (self.sample_weight*prob_ratios).exp())
        #print(f'edge compute loss: {time.time() - start_time}')
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        #ll, rr = map(lambda _x: np.array(_x), zip(*(samples.view(-1, 2)[final_decision].detach().cpu().numpy())))
        self.edge_perm[ll], self.edge_perm[rr] = self.edge_perm[rr].clone(), self.edge_perm[ll].clone()
        #print(f'edge change index: {time.time() - start_time}\n')
        return prob_ratios[final_decision].sum().item()

    
# Model that uses simple algorithm for finding the best permutation
class perm_model(kronecker_model):
    def __init__(self, graph, init_row, init_col, k, second_k, device, sample_weight, fixed):
        super().__init__(graph, init_row, init_col, k, second_k, device, sample_weight, fixed)
        self.matrix_row, self.matrix_col, _ = map(np.array, (self.graph.row_idx, self.graph.col_idx, self.graph.val))    
        
    '''
        Sample node permutation
        batch_size: batch size for nonzeros        
    '''
    def sample_node_batches(self, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: node idx -> pair idx                
            mapping: node idx -> node idx
        '''
        start_time = time.time()
        sampled_n = self.num_row
        # target_digit = 1 << int(random.random() * self.k)
        target_digit = np.random.randint(self.num_row)
        target_digit = target_digit & (-target_digit)
        if target_digit == 0: target_digit = self.num_row // 2
            
        sampled_kro_rows = torch.randperm(self.num_row // 2).to(self.i_device)
        sampled_kro_rows = (sampled_kro_rows * 2) - (sampled_kro_rows & (target_digit - 1)) + (target_digit * (torch.rand(self.num_row // 2).to(self.i_device) < 0.5).long())        
        
        perm_inv = torch.empty_like(self.node_perm)
        perm_inv[self.node_perm] = torch.arange(self.num_row).to(self.i_device)
        
        sampled_rows = perm_inv[sampled_kro_rows]  # idx -> row
        sampled_rows_mate = perm_inv[sampled_kro_rows^target_digit]      
        #print(f'node prepare min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        h_fn = torch.randperm(self.num_col).to(self.i_device) + 1        
        maxh = torch_scatter.scatter(h_fn[self.matrix_col], torch.LongTensor(self.matrix_row).to(self.i_device), dim=-1, dim_size=self.num_row, reduce='max')         
        maxh = maxh[sampled_rows]
        #print(f'node min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        maxh_last = torch_scatter.scatter(torch.arange(self.num_row // 2).to(self.i_device), maxh, dim=-1, dim_size=self.num_col + 1, reduce='max')
        maxh_count = torch.bincount(maxh, minlength=self.num_col + 1)
        odd_elements = maxh_last[(maxh_count % 2) > 0]
        maxh[odd_elements] = (self.num_col + 1)
        maxh = maxh * (self.num_row//2) + torch.arange(self.num_row//2).to(self.i_device)
        
        unmatched = torch.cat((sampled_rows[odd_elements], sampled_rows_mate[odd_elements]), dim=0)
        unmatched_n = unmatched.shape[-1]
        unmatched = unmatched[torch.randperm(unmatched_n)]
        #print(f'node handle remaining: {time.time() - start_time}') 
        
        # maxh: idx -> hash value
        # sorted_idx: new idx -> idx
        start_time = time.time()
        _, sorted_idx = torch.sort(maxh)
        # samples: new idx -> rows
        samples = torch.zeros(self.num_row, dtype=torch.long).to(self.i_device)
        samples[0::4], samples[1::4] = sampled_rows[sorted_idx[0::2]], sampled_rows_mate[sorted_idx[1::2]]
        samples[2::4], samples[3::4] = sampled_rows[sorted_idx[1::2]], sampled_rows_mate[sorted_idx[0::2]]
        samples = torch.cat((samples[:-unmatched_n], unmatched), dim=0)
        
        pair_idx = torch.arange(self.num_row).to(self.i_device)
        pair_idx[samples] = torch.arange(len(samples)).to(self.i_device)
        pair_idx //= 2
        mapping = torch.arange(len(samples)).to(self.i_device)
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
        sampled_points = torch.rand(self.num_row // 2, dtype=curr_dtype).to(self.i_device)
        prob_before = torch.zeros((self.num_row // 2), dtype=curr_dtype).to(self.i_device)
        prob_after = torch.zeros((self.num_row // 2), dtype=curr_dtype).to(self.i_device)

        # Compute the change of loss                                
        # non zero part (No need to consider zero part because prob_before and prob_after are the same when all entries are zero)
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            # Build an input for the current batch
            curr_batch_size = min(batch_size, num_nnz - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                   
            cols = self.edge_perm[edges]
            col_idx = cols.unsqueeze(1) // self.edge_bases % self.init_col
            row_idx_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
            curr_pair = pair_idx[nodes]
            curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
            
            # Compute prob_before & prob_after                                   
            temp_list = [(self.node_perm[nodes], prob_before), (self.node_perm[mapping[nodes]], prob_after)]
            for (curr_rows, curr_prob) in temp_list:                                                                              
                row_idx = curr_rows.unsqueeze(1) // self.node_bases % self.init_row                                        
                row_idx = torch.cat((row_idx, row_idx_pad), dim=-1)            
                inputs = row_idx * self.init_col + col_idx         
                outputs = self.model(inputs)             
                # Why the sign is like this?
                # Because metropolis-hasting use probability, decrease of loss means the increase of prob (-loss)
                curr_prob.index_add_(0, curr_pair, torch.square(outputs) - torch.square(outputs - curr_val))                                                                    

        prob_ratios = prob_after - prob_before
        prob_ratios.clamp_(min=-(10**15))    
        prob_thre = prob_ratios.clone()
        final_decision = (prob_thre >= 0.) | (sampled_points <= (self.sample_weight * prob_thre).exp())
        #print(f'node compute loss: {time.time() - start_time}')
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        #ll, rr = map(lambda _x: np.array(_x), zip(*(samples.view(-1, 2)[final_decision].detach().cpu().numpy())))
        self.node_perm[ll], self.node_perm[rr] = self.node_perm[rr].clone(), self.node_perm[ll].clone()     
        #print(f'node change index: {time.time() - start_time}\n')
        return prob_ratios[final_decision].sum().item()
    
    '''
        Sample edge permutation
        batch_size: batch size for nonzeros       
    '''
    def sample_edge_batches(self, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: edge idx -> pair idx
            mapping: edge idx -> edge idx
        '''
        start_time = time.time()
        sampled_n = self.num_col
        target_digit = np.random.randint(self.num_col)
        target_digit = target_digit & (-target_digit)
        if target_digit == 0: target_digit = self.num_col // 2
                
        sampled_kro_cols = torch.randperm(self.num_col // 2).to(self.i_device)
        sampled_kro_cols = (sampled_kro_cols * 2) - (sampled_kro_cols & (target_digit - 1)) + (target_digit * (torch.rand(self.num_col // 2).to(self.i_device) < 0.5).long())
        sampled_kro_cols = sampled_kro_cols[:(sampled_n // 2)]
        
        perm_inv = torch.empty_like(self.edge_perm)
        perm_inv[self.edge_perm] = torch.arange(self.num_col).to(self.i_device)
        
        sampled_cols = perm_inv[sampled_kro_cols]
        sampled_cols_mate = perm_inv[sampled_kro_cols^target_digit]        
        #print(f'edge prepare min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        h_fn = torch.randperm(self.num_row).to(self.i_device) + 1
        maxh = torch_scatter.scatter(h_fn[self.matrix_row], torch.LongTensor(self.matrix_col).to(self.i_device), dim=-1, dim_size=self.num_col, reduce='max')        
        maxh = maxh[sampled_cols]   # maxh: idx -> hash value
        #print(f'edge hashing: {time.time() - start_time}') 
            
        start_time = time.time()
        maxh_last = torch_scatter.scatter(torch.arange(self.num_col // 2).to(self.i_device), maxh, dim=-1, dim_size=self.num_row + 1, reduce='max')
        # maxh_last: hash value -> max idx
        maxh_count = torch.bincount(maxh, minlength=self.num_row + 1)
        # maxh_count: hash value -> count
        odd_elements = maxh_last[(maxh_count % 2) > 0] # odd_elements: new idx -> max idx where the count for the hash value is odd        
        maxh[odd_elements] = (self.num_row + 1)  
        maxh = maxh * (self.num_col//2) + torch.arange(self.num_col//2).to(self.i_device)        
        
        unmatched = torch.cat((sampled_cols[odd_elements], sampled_cols_mate[odd_elements]), dim=0)
        unmatched_n = unmatched.shape[-1]        
        unmatched = unmatched[torch.randperm(unmatched_n)]
        #print(f'edge handle remaining: {time.time() - start_time}') 
        
        # sorted_idx: new idx -> idx
        staart_time = time.time()
        _, sorted_idx = torch.sort(maxh)        
        # samples: new idx -> cols
        samples = torch.zeros(self.num_col, dtype=torch.long).to(self.i_device)
        samples[0::4], samples[1::4] = sampled_cols[sorted_idx[0::2]], sampled_cols_mate[sorted_idx[1::2]]
        samples[2::4], samples[3::4] = sampled_cols[sorted_idx[1::2]], sampled_cols_mate[sorted_idx[0::2]]
        samples = torch.cat((samples[:-unmatched_n], unmatched), dim=0)      
       
        pair_idx = samples.clone()
        pair_idx[samples] = torch.arange(len(samples)).to(self.i_device)
        pair_idx //= 2
        mapping = torch.arange(len(samples)).to(self.i_device)
        mapping[samples] = torch.stack((samples[1::2], samples[0::2]), dim=1).view(-1)
        #print(f'edge build mapping: {time.time() - start_time}')     

        '''
            sampled_points: pair idx -> sampled p
            prob_ratios: pair idx -> prob changes of the pair
        '''
        start_time = time.time()
        if self.data_type == "double": curr_dtype = torch.double                       
        elif self.data_type == "float": curr_dtype = torch.float        
        else: assert(False)        
        sampled_points = torch.rand(self.num_col // 2, dtype=curr_dtype).to(self.i_device)
        prob_ratios = torch.zeros((self.num_col // 2), dtype=curr_dtype).to(self.i_device)
        lb = -(10**9)

        # Compute the change of loss                                
        # non zero part
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            curr_batch_size = min(batch_size, num_nnz - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.device[0]), torch.LongTensor(edges).to(self.device[0])               
            curr_row = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
            curr_col = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
            curr_row_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.device[0])        
            curr_row = torch.cat((curr_row, curr_row_pad), dim=-1)
            curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.device[0])            
            
            neg_edges = mapping[edges]
            neg_col = self.edge_perm[neg_edges].unsqueeze(1) // self.edge_bases % self.init_col            

            inputs = torch.cat(((curr_row * self.init_col + curr_col), (curr_row * self.init_col + neg_col)), dim=0)                 
            outputs = self.model(inputs).view(2, -1)                      
            prob_ratios.index_add_(0, pair_idx[edges], (-torch.square(outputs[0]) + torch.square(outputs[0] - curr_val) - torch.square(outputs[1] - curr_val) + torch.square(outputs[1])))

        prob_ratios.clamp_(min=-(10**15))
        final_decision = (prob_ratios >= 0.) | (sampled_points <= (self.sample_weight*prob_ratios).exp())
        #print(f'edge compute loss: {time.time() - start_time}')
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        #ll, rr = map(lambda _x: np.array(_x), zip(*(samples.view(-1, 2)[final_decision].detach().cpu().numpy())))
        self.edge_perm[ll], self.edge_perm[rr] = self.edge_perm[rr].clone(), self.edge_perm[ll].clone()
        #print(f'edge change index: {time.time() - start_time}\n')
        return prob_ratios[final_decision].sum().item()

    '''
        Compute the L2 loss in an efficient manner   
        batch_size: batch size for nonzeros        
    '''
    def check_distribution(self, batch_size, save_file):        
        with torch.no_grad():
            # nnzs = list(zip(graph.adjcoo.row, graph.adjcoo.col))
            if len(self.device) > 1: sq_sum = (self.model.module.level_sq_sum)**(self.k + self.second_k)
            else: sq_sum = (self.model.level_sq_sum)**(self.k + self.second_k)
            sq_sum = sq_sum.item()   
            max_entry = max(self.graph.val)            
            loss_list = [0. for _ in range(max_entry + 1)]
            num_list = [0 for _ in range(max_entry + 1)]
            
            #for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):
            for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):    
                # Extract nodes and edges
                curr_batch_size = min(batch_size, self.graph.real_num_nonzero - i)
                nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
                nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                          
                curr_val = torch.LongTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)

                # Convert to lstm inputs
                row_idx = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
                col_idx = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
                row_idx_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
                row_idx = torch.cat((row_idx, row_idx_pad), dim=-1)

                # Correct non-zero parts
                inputs = row_idx * self.init_col + col_idx
                samples = self.model(inputs)
                #print(f'sample shape: {samples.shape}, value sahpe: {curr_val.shape}')      
                sq_sum -= torch.square(samples).sum().item()
                
                for i in range(1, max_entry + 1):
                    curr_idx = curr_val == i
                    loss_list[i] += torch.square(samples[curr_idx] - curr_val[curr_idx]).sum().item()
                    num_list[i] += curr_idx.long().sum().item()
                            
            loss_list[0] = sq_sum
            num_list[0] = self.num_row * self.num_col - self.graph.real_num_nonzero
            with open(save_file + ".txt", 'w') as f:
                for i in range(max_entry + 1):            
                    #print(f'{loss_list[i]}')
                    f.write(f'{num_list[i]}\t{loss_list[i]}\n')