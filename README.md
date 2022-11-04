# strang_splitting_hodgkin_huxley
Python implementation of the strang splitting solver for the hodgkin huxley neuron model as described in [Chen et al. (2020)](http://arxiv.org/abs/1811.00173).
Exploits the conditinally linear characteristic of the system of ODEs describing the Hodgkin Huxley model.

## Install
Tested on Ubuntu 20.04
### Conda
Tested on miniconda3 version 22.9.0
```
conda env create -f conda_env.yml
conda activate stranghh
```
### Venv
Tested on python version 3.8.10
```
python3 -m venv ~/venv/stranghh
source ~/venv/stranghh/bin/activate
pip install --upgrade pip setuptools
pip install -r pip_requirements.txt
```
