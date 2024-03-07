from itertools import chain
import torch
from scipy.sparse import coo_matrix
import numpy as np
import tqdm
import pickle


# Class that saves hypergraph
class hyperGraph:
    '''
        file_name: file that saves hypergraphs        
        allow_dupli: allow duplicated hyper-edges
    '''
    def getDegree(self):
        deg_list = self.inci_csr.sum(1)
        deg_list = np.squeeze(np.array(deg_list))       
        return list(deg_list)
    
    '''
        Extact edge indices for a given node index
    '''
    def extract_node(self, input_node):
        indptr = self.csr_matrix.indptr
        target_idx = self.csr_matrix.indices[indptr[input_node]:indptr[input_node + 1]]
        target_val = self.csr_matrix.data[indptr[input_node]:indptr[input_node + 1]]
        return target_idx, target_val
        
    '''
        Extract node indices for a given edge index
    '''
    def extract_edge(self, input_edge):
        indptr = self.inci_csc.indptr
        target_idx = self.inci_csc.indices[indptr[input_edge]:indptr[input_edge + 1]]
        target_val = self.inci_csc.data[indptr[input_edge]:indptr[input_edge + 1]]    
        return target_idx, target_val
    
    '''
        file_name: Name of the file
    '''
    def __init__(self, file_name):
        with open(file_name) as f:
            raw_data = f.read()
        
        lines = raw_data.split('\n')
        first_line = lines[0].split(',')
        self.num_row, self.num_col = int(first_line[0]), int(first_line[1])
        lines.pop(0)
        
        # Assume entres are unique
        lines = [[float(word) for word in line.split(",")] for line in lines if line]
        self.row_idx, self.col_idx, self.val = map(list, zip(*lines))
        self.row_idx = list(map(int, self.row_idx))
        self.col_idx = list(map(int, self.col_idx))
        
        self.entry_sum = sum(self.val)       
        self.sq_sum = sum([entry**2 for entry in self.val])
        print(f'entry sum: {self.entry_sum}, square sum: {self.sq_sum}')
        self.real_num_nonzero = len(self.val)        
        
        # Build matrix           
        self.inci_coo = coo_matrix((self.val, (self.row_idx, self.col_idx)), \
                       shape=(self.num_row, self.num_col))
        self.inci_csr, self.inci_csc = self.inci_coo.tocsr(), self.inci_coo.tocsc()       

        
# Class that saves hypergraph
class hyperTensor:
    def __init__(self, file_name):
        with open(file_name) as f:
            sizes = list(map(int, f.readline().split(',')))
            self.deg = sizes[0]
            self.dims = sizes[1:]
            print(f'# of dimension: {self.deg}, # of dims: {self.dims}')
            nnzs = [list(map(int, line.split(','))) for line in tqdm.tqdm(f)]
        
        self.real_num_nonzero = len(nnzs)
        self.indices = np.array(nnzs)
        # self.indices[:, :-1] -= 1
        self.indices, self.val = self.indices[:, :-1], self.indices[:, -1]
        self.entry_sum = self.val.sum()
        self.sq_sum = (self.val ** 2).sum()
        print(f'nnz: {self.real_num_nonzero}, entry sum: {self.entry_sum}, square sum: {self.sq_sum}')

        
class IrregularTensor:
    
    def __init__(self, file_path):
        '''
            Params for Neukron
        '''
        with open(file_path, 'rb') as f:
            raw_dict = pickle.load(f)            
       
        self.indices = np.transpose(np.array(raw_dict['idx']))
        self.real_num_nonzero = self.indices.shape[0]
        self.val = np.array(raw_dict['val'])
        self.deg = self.indices.shape[1]
        self.entry_sum = self.val.sum()
        self.sq_sum = (self.val ** 2).sum()
        print(f'nnz: {self.real_num_nonzero}, entry sum: {self.entry_sum}, square sum: {self.sq_sum}')
        
        '''
            Params for irregular tensor
        '''        
        idx2newidx = np.argsort(self.indices[:, -1])
        for m in range(self.deg):
            self.indices[:, m] = self.indices[idx2newidx, m]
        self.val = self.val[idx2newidx]            

        # save tensor stat       
        self.dims = []
        self.max_first = max(self.indices[:, 0]) + 1
        self.num_tensor = max(self.indices[:, -1]) + 1
        self.middle_dim = []
        for m in range(1, self.deg-1):
            self.middle_dim.append(max(self.indices[:, m]) + 1)                                           
        self.dims = [self.max_first] + self.middle_dim + [self.num_tensor]
     
        self.tidx2start = [0]
        for i in range(self.real_num_nonzero):
            if self.indices[self.tidx2start[-1], -1] != self.indices[i, -1]:
                self.tidx2start.append(i)
        self.tidx2start.append(self.real_num_nonzero)

        # Set first dim
        self.first_dim = []
        for i in range(self.num_tensor):
            self.first_dim.append(max(self.indices[self.tidx2start[i]:self.tidx2start[i+1], 0]) + 1)
        self.first_dim = np.array(self.first_dim)