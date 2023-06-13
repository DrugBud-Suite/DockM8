FROM python:3.8.10-slim
ARG username
ARG CUDA=cpu


# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*
    

# Create a working directory
RUN mkdir /main
RUN mkdir /main/home
WORKDIR /main


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash $username \
    && chown -R $username:$username /main
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER $username


# All users can use /home/user as their home directory
ENV HOME=/main/home
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

# Set up the Conda environment (using Miniforge)
ENV PATH=$HOME/mambaforge/bin:$PATH
COPY environment.yml /main/environment.yml
#COPY run.sh /main/home/run.sh
#RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-x86_64.sh 

RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-x86_64.sh \
    && chmod +x ~/mambaforge.sh  \
    && ~/mambaforge.sh -b -p ~/mambaforge 
    
RUN mamba env update -n base -f /main/environment.yml 
RUN python -m pip install torch==1.9.1 -f  https://data.pyg.org/whl/torch-1.9.0.html \
    && python -m pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html \
    && python -m pip install --upgrade setuptools \
    && python -m pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html \
    && python -m pip install torch-scatter==2.1.0  -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html\
    && python -m pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html 
    

    #&& rm /main/environment.yml \


#RUN rm ~/mambaforge.sh 
RUN ~/run.sh -b -p ~/run.sh

RUN mamba clean -ya

