# AEONS 

## Things to check:

- args for running aeons: device, host (localhost ?), port (9502 ?)
- basecaller config: "dna_r9.4.1_450bps_fast"  
- directions in readfish toml file
- location of masks and contigs in toml file



## Installation:


`mamba env create -f aeons_env_min.yaml`

```
conda activate aeons_env
pip install git+https://github.com/nanoporetech/read_until_api@v3.0.0
pip install git+https://github.com/W-L/readfish@issue208
apt list --installed ont-guppy* | tail -n 1 | cut -f2 -d' ' | cut -f1 -d'-' >guppy_version
pip install ont_pyguppy_client_lib==$(cat guppy_version)
git clone https://github.com/W-L/Aeons.git
```



## Run:


`readfish boss-runs --device X1 --experiment-name "zymolog001" --toml zymolog001.toml --port 9502 --log-file readfish_zymolog001.log` 


`python /Aeons/aeons_live.py @zymolog001.params &>aeons_zymolog001.log`



### CL for creating conda env 


```
conda create -n aeons_env
conda activate aeons_env
mamba install python=3.8 numpy==1.17.4 mappy minimap2 gfatools pandas toml pyfastx scipy
pip install ont_fast5_api

conda env export >aeons_env.yaml
```