#FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
#FROM ubuntu:22.04
FROM mambaorg/micromamba

#RUN apt update -y && apt autoremove -y
#RUN DEBIAN_FRONTEND=noninteractive apt install -y python3.11 python3.11-venv vim git curl --no-install-recommends
#RUN apt install -y vim git --no-install-recommends
USER root
RUN apt update -y && apt install -y git --no-install-recommends
#RUN DEBIAN_FRONTEND=noninteractive TZ="Europe/Berlin" apt install -y tzdata
RUN DEBIAN_FRONTEND=noninteractive apt autoremove
USER $MAMBA_USER
#RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
#RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
# installing pip packages uses /docker as tmp directory so mamba user needs to own it
ADD --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /work/
RUN /usr/bin/micromamba env create --file=/work/environment.yaml && /usr/bin/micromamba clean --all --yes
RUN cd /docker && git clone https://github.com/m5imunovic/fictional-pancake --depth=1
RUN /usr/bin/micromamba clean --all --yes

ENV LC_ALL=C
ENV PYTHONPATH=/work/fictional-pancake/src
ENV PROJECT_ROOT=/work/fictional-pancake
ENV WANDB_DIR=/work/data
ENV PATH="/root/miniforge3/bin:$PATH"

CMD ["/bin/bash"]
