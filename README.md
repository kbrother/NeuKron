# NeuKron: Constant-Size Lossy Compression of Sparse Reorderable Matrices and Tensors

This repository is the official implementation of NeuKron: Constant-Size Lossy Compression of Sparse Reorderable Matrices and Tensors (anonymized).

## Requirements

To run the provided codes, you need to install `PyTorch` and `PyTorch Scatter`. Since the installation commands for the packages rely on the environments, visit following pages to install the packages:

- `PyTorch`: https://pytorch.org/get-started/locally/
- `PyTorch Scatter`: https://github.com/rusty1s/pytorch_scatter

## Input file format

The format of an input file is as follows. For the first line, the order of tensor and the dimensions of each order is provided.

## Real-world datasets we used

|Name|shape|# of nonzeros|Source|Link|
|-|-|-|-|-|
|email|1,005 x 25,919|92,159|[Austin R. Benson](https://www.cs.cornell.edu/~arb/data/email-Eu/)|[Link](https://drive.google.com/open?id=1oFnrRaLPlYCNgABInjjW8fUgui49eOZN)|
|nyc|1,083 x 38,333|91,024|||
|tky|2,293 x 61,858|211,955|||
|kasandr|414,520 x 503,702|903,366|||
|threads|176,445 x 595,778|1,457,727|[Austin R. Benson](https://www.cs.cornell.edu/~arb/data/threads-math-sx/)|[Link](https://drive.google.com/open?id=1aoNCO5IfY14cIKyTir-qAZl78sgMixhA)|
|twitch|790,100 x 15,524,309|234,422,289|[Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch)|[Link](https://www.google.com/url?q=https://drive.google.com/drive/folders/1BD8m7a8m7onaifZay05yYjaLxyVV40si?usp%3Dsharing&source=gmail-html&ust=1629428377164000&usg=AFQjCNFXimN1hHftvhgIu5iKUTZiOsep8A)|
|nips|2,482 x 2,862 x 14,036|3,101,609|[FROSTT](http://frostt.io/tensors/nips/)|[Link](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz)|
|4-gram|48,419 x 53,654 x 54,801 x 58,043|7,495,550|[N-grams data](https://www.ngrams.info/download_coha.asp)|[Link](https://www.ngrams.info/download_coha.asp)|
|3-gram|87,614 x 100,170 x 109,721|9,778,281|[N-grams data](https://www.ngrams.info/download_coha.asp)|[Link](https://www.ngrams.info/download_coha.asp)|
|enron|5,699 x 6,066 x 244,268|31,312,375|[FROSTT](http://frostt.io/tensors/enron/)|[Link](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/enron/enron.tns.gz)|
