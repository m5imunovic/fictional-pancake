# GnnDebugger

## Installation and development

We recommend the usage of mambaforge for setting up Python environment:

https://github.com/conda-forge/miniforge#mambaforge

```Bash
### Install conda
mamba init

### Create new environment (named menv)
mamba env create --file=environment.yaml
mamba activate menv
```

### Setup pre-commit hooks

```bash
pip install pre-commit
# Set up on the first usage
pre-commit install
pre-commit run --all-files
```

## Training

For training, we recommend setting up singularity image, script assume that the project is > \[!IMPORTANT\]
`~/work/gnndebugger`

```Bash
cd apptainer && bash build_apptainer.sh
```

Configs should be modified according to the local paths (check `config/paths/paths.yaml`)
Default data directory is `/data`.
Datasets are expected in `/data/datasets` directory.
Place configuration in `/data/configs` directory.

This will build image `dbgc.sif` which can then be used for training. Place the training

```Bash
apptainer run \
        --bind $HOME/data:/data \
        --nv $HOME/work/gnndebugger/dbgc.sif \
        --config-path /data/config \
        --config-name "train.yaml" \
```
