# AEONS 


## Requirements

- Linux
- conda/mamba/micromamba
- MinKNOW >=5.0 (with guppy >=6.0 or newer dorado)


## Installation


This software runs readfish in the background, therefore first install readfish. This is based on instructions found at [readfish - repo](https://github.com/LooseLab/readfish/tree/main#installation)
AEONS runs in the same environment after installing a few additional dependencies below.

```shell
cat <<EOT > readfish-aeons.yml
name: readfish
channels:
  - bioconda
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - readfish[all]
EOT
```

```shell
# create environment for readfish and AEONS
mamba env create -f readfish-aeons.yml -y
mamba activate readfish
# additional dependencies for AEONS
mamba install -y gfatools minimap2 miniasm bottleneck -c bioconda -c conda-forge -c defaults
# finally clone this repository
git clone https://github.com/W-L/Aeons.git
```


## Usage

To configure a sequencing experiment we need 2 separate TOML files: one for AEONS, one for readfish.

AEONS consequently only takes two arguments pointing to those two TOML files:

`python ./aeons.py --toml [aeons.toml] --toml_readfish [readfish.toml]`

For AEONS, here is a template with all default values: (Modify/delete this template to configure an experiment.)

```
[general]
name = "aeons"                   # experiment name

[live]
device = ""                     # position on sequencer
host = "localhost"              # host of sequencing device
port = 9502                     # port of sequencing device
data_wait = 100                 # wait for X Mb data before first asm
prom = false                    # switch for promethION flowcell

[optional]
temperature = 60                # max batches during which to consider fragments
wait = 60                       # waiting time between updates in live version

[filters]                       # overwrite filters
min_seq_len = 2500
min_contig_len = 10_000
min_s1 = 200
min_map_len = 2000

[const]                         # overwrite constants
mu = 400                        # length of anchor bases
tetra = true                    # perform tetranucleotide frequency tests
filter_repeats = false          # perform repeat filtering
lowcov = 10                     # target for assemblies

[simulation]                    # simulation arguments
fq = ""
bsize = 4000                    
maxb = 400
binit = 5
dumptime = 200000000         
```

For configuration of readfish TOML files, please see: [readfish - repo - toml](https://github.com/LooseLab/readfish/blob/main/docs/toml.md)



## Walkthrough and testing

...


### Unittests

```
pip install coverage
coverage run -m unittest discover -s tests
coverage report
```

## Issues, questions, suggestions ...

...

## Citation

...

## License

Licensed under GPLv3



