#!/bin/bash
set -o xtrace

G2G_REPO_PREFIX="https://github.com/abojchevski/graph2gauss/raw/master/data"
GAUG_REPO_PREFIX="https://github.com/zhao-tong/GAug/raw/master/data/graphs"

for dataset in citeseer cora cora_ml dblp pubmed
do
	curl -L --create-dirs -o ./data/${dataset}/raw/${dataset}.npz ${G2G_REPO_PREFIX}/${dataset}.npz
done


for dataset in airport blogcatalog flickr
do
	for suffix in adj features labels tvt_nids
	do
		curl -L --create-dirs -o ./data/alternative/${dataset}_${suffix}.pkl ${GAUG_REPO_PREFIX}/${dataset}_${suffix}.pkl
	done
done