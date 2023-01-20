# Scale-Free, Attributed and Class-Assortative Graph Generation

This repository contains code for the paper ["Scale-Free, Attributed and Class-Assortative Graph Generation to Facilitate Introspection of Graph Neural Networks,"](https://nshah.net/publications/CABAM.MLG.2020.pdf) published at KDD MLG 2020.  In particular, it enables use of the CABAM (Class Assortative Barabasi Albert Model) for graph generation, which enhances the traditional Barabasi-Albert preferential attachment model with flexible class labels, feature distributions and assortativity (homophily) parameters.

## Installation

You can install a `cabam` package with 

```
pip install .
```

## Downloading Data

You can see `./Dataset Summaries.ipynb` for reproducing dataset statistics mentioned in the paper.

Data referenced in this code is sourced from versions available at 
[graph2gauss](https://github.com/abojchevski/graph2gauss) and 
[GAug](https://github.com/zhao-tong/GAug) repositories.  

Please run:

```
bash download_data.sh
```

in order to download all relevant data files referenced in scripts here (all files will be placed under `./data`), before trying to run notebook commands yourself.

## Generating graphs with CABAM

The below is a basic example, to use CABAM to generate a graph with with 1000 nodes and minimum degree of 3, 
with two balanced classes (default) and roughly equal numbers of homophilous and heterophilous edges (default).

```
from cabam import CABAM
model = CABAM()
G, degrees, labels, num_intra_edges, num_inter_edges = model.generate_graph(n=1000, m=3)
```

See `generate_graph(...)` in `./cabam/cabam.py` for customizing use.

## Reference

If you use the model, or graphs generated with the model for evaluation in your own work, please cite

```BibTeX
 @inproceedings{cabam2020shah,
     author = {Shah, Neil},
     title = {Scale-Free, Attributed and Class-Assortative Graph Generation to Facilitate Introspection of Graph Neural Networks},
     booktitle = {KDD Mining and Learning with Graphs},
     year = {2020}
   }
```
