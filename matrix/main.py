import sys
import torch
import argparse
from graph import hyperGraph
from model import perm_model, kronecker_model
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

def eval_loss(k_model, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    checkpoint = torch.load(args.load_path + ".pt", map_location=device)
    k_model.model.load_state_dict(checkpoint['model_state_dict'])
    k_model.node_perm = checkpoint['node_perm']
    k_model.edge_perm = checkpoint['edge_perm']
    
    k_model.model.eval()
    with torch.no_grad():
        print(f'Approximation Error:{k_model.L2_loss(False, args.batch_size)}')

def train_model(k_model, args):
    print(f'learning rate: {args.lr}')
    curr_lr = args.lr
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    print(f'learning rate: {args.lr}')    
    optimizer = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    if args.retrain:
        # Load paramter and optimizer
        checkpoint = torch.load(args.load_path + "_min.pt", map_location=device)
        k_model.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['loss']
        k_model.node_perm = checkpoint['node_perm']
        k_model.edge_perm = checkpoint['edge_perm']
    else:
        print("RETRAIN=False")
        start_epoch = 0
        min_loss = sys.float_info.max  
        
    epoch = 0
    for epoch in range(start_epoch, args.max_epochs):                                                                   
        # Sample permutation         
        time_per_epoch = 0.
        start_time = time.time()
        k_model.model.eval()
        with torch.no_grad():
            for _ in range(args.perm_per_update):
                k_model.sample_node_batches(args.batch_size)
                k_model.sample_edge_batches(args.batch_size)                                                      
        
        # Update parameter 
        k_model.model.train()
        optimizer.zero_grad()
        perm_loss = k_model.L2_loss(True, args.batch_size)                            
        with open(args.save_path + ".txt", 'a') as lossfile:   
            lossfile.write(f'epoch:{epoch}, loss after perm:{perm_loss}\n')    
            print(f'epoch:{epoch}, loss after perm:{perm_loss}\n')
            
        #scheduler.step(training_loss)
        # Back-up parameters                
        prev_opt_params = copy.deepcopy(optimizer.state_dict())       
        prev_params = copy.deepcopy(k_model.model.state_dict())
        optimizer.step()    
        
        k_model.model.eval()
        with torch.no_grad():            
            model_loss = k_model.L2_loss(False, args.batch_size)           
                    
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
                                        
        time_per_epoch = time.time() - start_time        
        with open(args.save_path + ".txt", 'a') as lossfile:                
            lossfile.write(f'epoch:{epoch}, loss after model update:{model_loss}, time per epoch: {time_per_epoch}\n')
        print(f'epoch:{epoch}, loss after model update:{model_loss}, time after model: {time_per_epoch}\n')

        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': k_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model_loss,
                'node_perm': k_model.node_perm,
                'edge_perm': k_model.edge_perm
            }, args.save_path + ".pt")                


        if min_loss > model_loss:
            min_loss = model_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': k_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model_loss,
                'node_perm': k_model.node_perm,
                'edge_perm': k_model.edge_perm
            }, args.save_path + "_min.pt") 

def load_model(k_model, args, device):
    # Load model
    use_cuda = torch.cuda.is_available()
    i_device = torch.device("cuda:" + str(device[0]) if use_cuda else "cpu")  
    checkpoint = torch.load(args.load_path + ".pt", map_location=i_device) # , map_location=args.device
    k_model.model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    k_model.node_perm = checkpoint['node_perm']
    k_model.edge_perm = checkpoint['edge_perm']
    k_model.model.eval()
    
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train, eval')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument(
        "-td", "--test_data", 
        action="store", default="none", type=str
    )
    parser.add_argument(
        "-r", "--init_row",
        action="store", default=2, type=int
    )
    
    parser.add_argument(
        "-p", "--perm_file",
        action="store", default=False, type=bool
    )
    
    parser.add_argument(
        "-c", "--init_col",
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
        action="store", default=10**5, type=int
    )
    
    parser.add_argument(
        "-lr", "--lr",
        action="store", default=1e-1, type=float
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
        action="store", default="False", type=str
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
    data_file = "../data/" + args.dataset + ".txt"    
    hgraph = hyperGraph(data_file)    
    print(f'rows: {hgraph.num_row}, columns:{hgraph.num_col}, nnz:{hgraph.real_num_nonzero}')
    
    # Initialize model
    k, second_k = 1, 1    
    curr_row, curr_col = args.init_row, args.init_col
    while curr_row < hgraph.num_row:
        k += 1 
        curr_row *= 2        
    while curr_col < hgraph.num_col:
        second_k += 1
        curr_col *= 2
    second_k -= k
        
    print(f'k: {k}, second k: {second_k}')
    if args.use_min == "True":
        k_model = perm_model(hgraph, args.init_row, args.init_col, k, second_k, args.device, args.sample_weight, args.fixed) 
    elif args.use_min == "False":
        k_model = kronecker_model(hgraph, args.init_row, args.init_col, k, second_k, args.device, args.sample_weight, args.fixed) 
    else:
        assert(False)
        
    if args.load_perm == "True":
        row_perm_file, col_perm_file = "../data/" + args.dataset + "_row.txt", "../data/" + args.dataset + "_col.txt" 
        k_model.set_permutation(row_perm_file, col_perm_file)
    else:
        k_model.init_permutation()
        
    input_size, input_size_sec = args.init_row * args.init_col, args.init_col
    k_model.init_model(args.hidden_size, args.model, args.data_type, second_k > 0)
    
    # test loss
    if args.action == 'train':
        train_model(k_model, args) 
    elif args.action == 'eval':
        eval_loss(k_model, args)
    else:
        assert(False)