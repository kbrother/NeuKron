import sys
import torch
torch.set_num_threads(4)

import argparse
from graph import hyperGraph, hyperTensor, IrregularTensor
from model import perm_model, kronecker_model
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from tqdm import tqdm
import math

def eval_loss(k_model, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    checkpoint = torch.load(args.load_path + ".pt", map_location=device)
    k_model.model.load_state_dict(checkpoint['model_state_dict'])
    k_model.perms = checkpoint['perms']
    
    k_model.model.eval()
    with torch.no_grad():
        print(f'Approximation Error: {k_model.L2_loss(False, args.batch_size)}')


def train_model(k_model, args):
    print(f'learning rate: {args.lr}')
    curr_lr = args.lr
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    optimizer = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    if args.retrain:
        # Load paramter and optimizer
        checkpoint = torch.load(args.load_path + "_min.pt", map_location=device)
        k_model.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['loss']
        k_model.perms = checkpoint['perms']
    else:
        print("RETRAIN=False")
        start_epoch = 0
        min_loss = sys.float_info.max  
        
    epoch = 0
    _end_counter = 0
    start_time = time.time()
    for epoch in range(start_epoch, args.max_epochs):                                                                   
        # Sample permutation         
        start_time = time.time()
        k_model.model.eval()
        with torch.no_grad():
            for _ in range(args.perm_per_update):
                if args.use_min == "True":
                    k_model.update_hashs()
                for i in range(k_model.order):
                    k_model.update_perm(i, args.batch_size)        
        
        # Update parameter 
        k_model.model.train()
        optimizer.zero_grad()
        perm_loss = k_model.L2_loss(True, args.batch_size)                            
        #with open(args.save_path + ".txt", 'a') as lossfile:               
        #    lossfile.write(f'epoch:{epoch}, loss after perm:{perm_loss}\n')    
        #    print(f'epoch:{epoch}, loss after perm:{perm_loss}\n')
        
        #scheduler.step(training_loss)
        # Back-up parameters                
        prev_opt_params = copy.deepcopy(optimizer.state_dict())       
        prev_params = copy.deepcopy(k_model.model.state_dict())
        optimizer.step()    
        
        k_model.model.eval()
        with torch.no_grad():            
            model_loss = k_model.L2_loss(False, args.batch_size)           
            # print(model_loss, k_model.loss_naive(args.batch_size))
            
        # Loss increased a lot      
        _cnt = 0
        while epoch > 0 and model_loss > 1.1*perm_loss:
            _cnt += 1
            # Restore parameters                        
            k_model.model.load_state_dict(prev_params)
            optimizer.load_state_dict(prev_opt_params)
            for i, param_group in enumerate(optimizer.param_groups):                
                param_group['lr'] = 0.1 * float(param_group['lr'])
             
            # Update parameter                            
            optimizer.step()                
            k_model.model.eval()
            with torch.no_grad():            
                model_loss = k_model.L2_loss(False, args.batch_size)           
        
            if model_loss < 1.1*perm_loss or _cnt == 10:
                for i, param_group in enumerate(optimizer.param_groups):                
                    param_group['lr'] = args.lr
                break
                                        
        time_elapsed = time.time() - start_time        
        curr_fit = 1 - math.sqrt(model_loss/k_model.graph.sq_sum)
        with open(args.save_path + ".txt", 'a') as lossfile:              
            lossfile.write(f'epoch:{epoch}, Fitness:{curr_fit}, running time: {time_elapsed}, end_counter: {_end_counter}\n')
        print(f'epoch:{epoch}, Fitness:{curr_fit}, running time: {time_elapsed}\n')
        
        if (min_loss - model_loss) / min_loss <= 1e-5:
            _end_counter += 1
        else:
            _end_counter = 0
            
        if min_loss > model_loss:
            min_loss = model_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': k_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model_loss,
                'perms': k_model.perms
            }, args.save_path + "_min.pt") 
        if _end_counter >= 100:
            break
                
def noperm_train(k_model, args):
    print(f'learning rate: {args.lr}')
    curr_lr = args.lr
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    print(f'learning rate: {args.lr}')    
    optimizer = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    print("RETRAIN=False")
    start_epoch = 0
    min_loss = sys.float_info.max  
        
    epoch = 0
    prev_time_model, curr_time_model = 0., 0.
    for epoch in range(start_epoch, args.max_epochs):                                                                           
        prev_time_model = curr_time_model
        # Update parameter 
        start_time = time.time()
        k_model.model.train()
        optimizer.zero_grad()
        prev_loss = k_model.L2_loss(True, args.batch_size)  
        prev_params = copy.deepcopy(k_model.model.state_dict())
        prev_opt_params = copy.deepcopy(optimizer.state_dict())
        
        optimizer.step()        
        curr_time_model = time.time() - start_time        
        
        if epoch > 0:
            with open(args.save_path + ".txt", 'a') as lossfile:                
                lossfile.write(f'epoch:{epoch}, loss after model update:{prev_loss}, time after model: {prev_time_model}\n')
            print(f'epoch:{epoch}, loss after model update:{prev_loss}, time after model: {prev_time_model}\n')
        
            if epoch % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': prev_params,
                    'optimizer_state_dict': prev_opt_params,
                    'loss': prev_loss,                    
                }, args.save_path + ".pt")                


            if min_loss > prev_loss:
                min_loss = prev_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': prev_params,
                    'optimizer_state_dict': prev_opt_params,
                    'loss': prev_loss,                    
                }, args.save_path + "_min.pt") 
    
    with torch.no_grad():
        k_model.model.eval()
        curr_loss = k_model.L2_loss(False, args.batch_size)  
        
    with open(args.save_path + ".txt", 'a') as lossfile:                
        lossfile.write(f'epoch:{epoch}, loss after model update:{curr_loss}, time after model: {curr_time_model}\n')
        print(f'epoch:{epoch}, loss after model update:{curr_loss}, time after model: {curr_time_model}\n')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': k_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': curr_loss,                    
        }, args.save_path + ".pt")                

        if min_loss > curr_loss:            
            torch.save({
                'epoch': epoch,
                'model_state_dict': k_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': curr_loss,                    
            }, args.save_path + "_min.pt") 

def load_model(k_model, args, device):
    # Load model
    checkpoint = torch.load(args.load_path + ".pt", map_location=device) # , map_location=args.device
    k_model.model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    k_model.perms = checkpoint['perms']
    
    # Check load is succeed
    k_model.model.eval()
              
        
# python tensor/main.py -d cms -de 0 -hs 10 -b 262144 -e 500 -lr 1e-2 -sp results/cms_hs10_lr0.01 -dt double 
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train, eval')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument(
        "-sz", "--init_size",
        action="store", default=2, type=int
    )
    
    parser.add_argument(
        "-de", "--device",
        action="store", nargs='+', type=int
    )
    
    parser.add_argument(
        "-hs", "--hidden_size",
        action="store", default=11, type=int
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        action="store", default=2**18, type=int
    )
    
    parser.add_argument(
        "-e", "--max_epochs",
        action="store", default=500, type=int
    )
    
    parser.add_argument(
        "-lr", "--lr",
        action="store", default=1e-2, type=float
    )
    
    parser.add_argument(
        "-lp", "--load_path",
        action="store", default="./params/", type=str
    )
    
    parser.add_argument(
        "-sp", "--save_path",
        action="store", default="./params/", type=str
    )
    
    parser.add_argument(
        "-rt", "--retrain",
        action="store", default=False, type=bool
    )
    
    parser.add_argument(
        "-perm", "--load_perm",
        action="store", default="True", type=str
    )
    
    parser.add_argument(
        "-ppu", "--perm_per_update",
        action="store", default=2, type=int
    )
    
    parser.add_argument(
        "-m", "--model",
        action="store", default="many2many", type=str
    )
    
    parser.add_argument(
        "-sw", "--sample_weight",
        action="store", default=10, type=float
    )
    
    parser.add_argument(
        "-f", "--fixed",
        action="store", default=False, type=bool
    )
    
    parser.add_argument(
        "-dt", "--data_type",
        action="store", default="float", type=str
    )    
    
    parser.add_argument(
        "-um", "--use_min",
        action="store", default="True", type=str
    )
    
    args = parser.parse_args()
    print(f'fixed:{args.fixed}, hidden state: {args.hidden_size}, perm: {args.load_perm}')
    
    # Load graph    
    data_file = "../data/23-Irregular-Tensor/" + args.dataset + ".pickle"    
    hgraph = IrregularTensor(data_file)    
    print(f'dims:{hgraph.dims}, nnz:{hgraph.real_num_nonzero}')
    
    # Initialize model
    ks = []
    for i in range(hgraph.deg):
        _dim = (hgraph.dims[i] - 1)
        _k = 0
        while _dim > 0:
            _dim, _k = _dim // 2, _k + 1
        ks.append(_k)
    print(f'ks: {ks}')
        
    if args.use_min == "False":
        k_model = kronecker_model(hgraph, args.init_size, ks, args.device, args.sample_weight, args.fixed)
    elif args.use_min == "True":
        k_model = perm_model(hgraph, args.init_size, ks, args.device, args.sample_weight, args.fixed) 
    else:
        assert(False)
    
    if (len(ks) == 2) and args.load_perm == "True":
        row_perm_file, col_perm_file = "../data/" + args.dataset + "_row.txt", "../data/" + args.dataset + "_col.txt" 
        k_model.set_permutation(row_perm_file, col_perm_file)
    else:
        if args.load_perm:
            print('load_perm => False (since len(ks) > 2)')
        k_model.init_permutation()
        
    k_model.init_model(args.hidden_size, args.model, args.data_type, args.save_path)
    
    # test loss
    if args.action == 'train':
        train_model(k_model, args)  
    elif args.action == 'eval':
        eval_loss(k_model, args)
    else:
        assert(False)