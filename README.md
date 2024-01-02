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
mamba install -y gfatools minimap2 bottleneck -c bioconda -c conda-forge -c defaults
# finally clone this repository
git clone https://github.com/W-L/Aeons.git
```


## Usage

To configure a sequencing experiment we need 2 separate TOML files: one for AEONS, one for readfish.

AEONS consequently only takes two arguments pointing to those two TOML files:

`python ./aeons.py --toml [aeons.toml] --toml_readfish [readfish.toml]`

For AEONS a template with all default values is included in `/aeons`. Modify/delete this template to configure an experiment.
For configuration of readfish TOML files, please see: [readfish - repo - toml](https://github.com/LooseLab/readfish/blob/main/docs/toml.md)



## Walkthrough and testing

...

## Issues, questions, suggestions ...

...

## Citation

...

## License

Licensed under GPLv3



