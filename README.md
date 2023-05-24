# AEONS 

## Things to check:

- check connection details: 
  - device (X1 - X5 ?) -> needs to be set in .params
  - host (localhost ?), port (9502 ?) -> set in .toml and on readfish commandline
- basecaller config: "dna_r9.4.1_450bps_fast"  




## Installation:

```
mamba env create -f aeons_env_synmix.yaml
conda activate aeons_env
git clone https://github.com/W-L/Aeons.git
```

## Run:

```
readfish boss-runs --device X1 --experiment-name "synmix" --toml synmix.toml --port 9502 --log-file readfish_synmix.log --chunk-log chunks.tsv --paf-log rf.paf

python /Aeons/aeons_live.py @synmix.params &>aeons_synmix.log
```




### CL for creating conda env 


```
conda create -n aeons_env
conda activate aeons_env
mamba install python=3.8 numpy==1.17.4 mappy minimap2 gfatools pandas toml pyfastx scipy bottleneck
pip install ont_fast5_api

conda env export >aeons_env.yaml
```