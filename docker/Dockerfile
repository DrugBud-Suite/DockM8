
FROM ubuntu:jammy
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
    

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash $username \
    && chown -R $username:$username /home
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER $username


# All users can use /home/user as their home directory
ENV HOME=/home
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

# Set up the Conda environment 
ENV PATH=$HOME/mambaforge/bin:$PATH
COPY environment.yml /home/environment.yml

RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-x86_64.sh \
    && chmod +x ~/mambaforge.sh  \
    && ~/mambaforge.sh -b -p /home/mambaforge 
    
RUN mamba env update -n base -f /home/environment.yml 
RUN pip install torch -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html \
    && pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html \
    && pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html \
    && pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html 

WORKDIR /home/$username

RUN git clone https://gitlab.com/Tonylac77/DockM8 \
    && rm /home/environment.yml \ 
    && rm ~/mambaforge.sh \ 
    && mamba clean -ya

