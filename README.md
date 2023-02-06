# NeuKron: Constant-Size Lossy Compression of Sparse Reorderable Matrices and Tensors

This repository is the official implementation of NeuKron: Constant-Size Lossy Compression of Sparse Reorderable Matrices and Tensors, Taehyung Kwon*, Jihoon Ko*, Jinghong jung, and Kijung Shin, WWW 2023.

## Requirements

To run the provided codes, you need to install `PyTorch` and `PyTorch Scatter`. Since the installation commands for the packages rely on the environments, visit following pages to install the packages:

- `PyTorch`: https://pytorch.org/get-started/locally/
- `PyTorch Scatter`: https://github.com/rusty1s/pytorch_scatter

## Arguments for training and evaluation

### Positional argument
- `action`: `train` for training the model, `eval` for evaluation

### Optional arguments (common)
- `-hs`, `--hidden_size`: hidden dimension of the model
- `-b`, `--batch_size`: batch size for training and evaluating the model
- `-de`, `--device`: gpu id(s) for execution

### Optional arguments (for training)
- `-e`, `--max_epochs`: number of training epochs
- `-lr`, `--lr`: learning rate for the parameters
- `-sp`, `--save_path`: path for saving the parameters
- `-ppu`, `--perm_per_update`: a number of permutation updates per epoch
- `-perm`: enable order initialization (matrix only)

### Optional arguments (for evaluation)
- `-lp`, `--load_path`: path for loading the parameters

## Example Commands

For example dataset, we provide ```example_matrix.txt``` and ```example_tensor.txt``` in `data/`.

To train the model, run
```
python main.py train -d [DATASET_NAME] -de 0 1 -hs 90 -lr 1e-2 -e 1500 --ppu 2 -sp [CHECKPOINT_PATH_FOR_SAVE]
```

To evaluate the model, run
```
python main.py eval -d [DATASET_NAME] -de 0 1 -hs 90 -lp [CHECKPOINT_PATH_FOR_LOAD]
```

## Real-world datasets we used

|Name|shape|# of nonzeros|Source|Link|
|-|-|-|-|-|
|email|1,005 x 25,919|92,159|[Austin R. Benson](https://www.cs.cornell.edu/~arb/data/email-Eu/)|[Link](https://drive.google.com/open?id=1oFnrRaLPlYCNgABInjjW8fUgui49eOZN)|
|nyc|1,083 x 38,333|91,024|[Dingqi Yang](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)|[Link](http://www.google.com/url?q=http%3A%2F%2Fwww-public.it-sudparis.eu%2F~zhang_da%2Fpub%2Fdataset_tsmc2014.zip&sa=D&sntz=1&usg=AFQjCNG04iQk2xN3SLlG7X4r1fGjb88Tpw)|
|tky|2,293 x 61,858|211,955|[Dingqi Yang](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)|[Link](http://www.google.com/url?q=http%3A%2F%2Fwww-public.it-sudparis.eu%2F~zhang_da%2Fpub%2Fdataset_tsmc2014.zip&sa=D&sntz=1&usg=AFQjCNG04iQk2xN3SLlG7X4r1fGjb88Tpw)|
|kasandr|414,520 x 503,702|903,366|[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/KASANDR)|[Link](https://archive.ics.uci.edu/ml/datasets/KASANDR)|
|threads|176,445 x 595,778|1,457,727|[Austin R. Benson](https://www.cs.cornell.edu/~arb/data/threads-math-sx/)|[Link](https://drive.google.com/open?id=1aoNCO5IfY14cIKyTir-qAZl78sgMixhA)|
|twitch|790,100 x 15,524,309|234,422,289|[Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch)|[Link](https://www.google.com/url?q=https://drive.google.com/drive/folders/1BD8m7a8m7onaifZay05yYjaLxyVV40si?usp%3Dsharing&source=gmail-html&ust=1629428377164000&usg=AFQjCNFXimN1hHftvhgIu5iKUTZiOsep8A)|
|nips|2,482 x 2,862 x 14,036|3,101,609|[FROSTT](http://frostt.io/tensors/nips/)|[Link](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz)|
|4-gram|48,419 x 53,654 x 54,801 x 58,043|7,495,550|[N-grams data](https://www.ngrams.info/download_coha.asp)|[Link](https://www.ngrams.info/download_coha.asp)|
|3-gram|87,614 x 100,170 x 109,721|9,778,281|[N-grams data](https://www.ngrams.info/download_coha.asp)|[Link](https://www.ngrams.info/download_coha.asp)|
|enron|5,699 x 6,066 x 244,268|31,312,375|[FROSTT](http://frostt.io/tensors/enron/)|[Link](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/enron/enron.tns.gz)|
