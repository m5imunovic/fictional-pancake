#FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
FROM ubuntu:22.04

RUN apt update -y && apt autoremove -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y curl git ca-certificates --no-install-recommends
#RUN DEBIAN_FRONTEND=noninteractive TZ="Europe/Berlin" apt install -y tzdata
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
COPY environment.yaml /work/
RUN /root/miniforge3/bin/mamba env create --file=/work/environment.yaml && /root/miniforge3/bin/mamba clean --all --yes
RUN cd /work && git clone https://github.com/m5imunovic/fictional-pancake --depth=1
RUN cd / && rm Miniforge3-$(uname)-$(uname -m).sh && apt remove git curl -y && apt autoremove -y

ENV LC_ALL=C
ENV PYTHONPATH=/work/fictional-pancake/src
ENV PROJECT_ROOT=/work/fictional-pancake
ENV WANDB_DIR=/work/data
ENV PATH="/root/miniforge3/bin:$PATH"

WORKDIR /work/fictional-pancake

ENTRYPOINT ["/root/miniforge3/bin/mamba", "run", "-n", "fenv", "python"]
#CMD ["/bin/bash"]
