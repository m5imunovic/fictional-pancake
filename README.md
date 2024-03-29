# fictional-pancake

# Installation

We recommend the usage of mambaforge for setting up Python environment:

https://github.com/conda-forge/miniforge#mambaforge

```Bash
## Install conda
mamba init

## Create new environment (named menv)
mamba env create --file=environment.yaml
mamba activate menv
```

## Setup pre-commit hooks

```bash
pip install pre-commit
# Set up on the first usage
pre-commit install
pre-commit run --all-files
```

## Run experiment

In order to run specific experiment override the experiment value accordingly:

```Bash
# Resgated model working with directed graph data
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_digraph
# Resgated model working with multidirected graph data
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph
# SIGN model working with directed graph data
PROJECT_ROOT=./ python src/train.py experiment=exp_sign_digraph
```

## Transform data

Usually, we want to transform data before the experiments. `dataset_name` option must match the name
of the dataset directory with graph data for transformation. Then you can run the following command:

```bash
# Transform multidigraph data using default transformations
PROJECT_ROOT=./ python src/transform.py graph=multidigraph dataset_name=random_species_10_01
```
