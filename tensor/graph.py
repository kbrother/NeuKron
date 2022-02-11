from itertools import chain
import torch
from scipy.sparse import coo_matrix
import numpy as np
import tqdm

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
        lines = [[int(float(word)) for word in line.split(",")] for line in lines if line]
        self.row_idx, self.col_idx, self.val = map(list, zip(*lines))
        
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
        