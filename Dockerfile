#FROM nvidia/cuda:11.2.0-devel-ubuntu18.04
FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

LABEL maintainer="Sergey Ustyantsev ustyantsev@gmail.com"

RUN apt-get clean && apt-get update && apt-get install -yqq curl && apt-get clean

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash

RUN ln -fs /usr/share/zoneinfo/Russia/Moscow /etc/localtime
ENV DEBIAN_FRONTEND noninteractive

# https://github.com/pyenv/pyenv/wiki#suggested-build-environment see at the required dependencies for pyenv
RUN apt-get install -yqq build-essential cmake cuda-toolkit-11-0 curl gfortran git graphviz htop libatlas-base-dev \
        libatlas3-base libblas-dev libbz2-dev libcudnn8 libffi-dev libfreetype6-dev libhdf5-dev liblapack-dev \
        liblapacke-dev liblzma-dev libncurses5-dev libpng-dev libreadline-dev libsqlite3-dev \
        libssl-dev libxml2-dev libxmlsec1-dev libxslt-dev llvm locales make nano nodejs pkg-config \
        tk-dev tmux tzdata wget xz-utils zlib1g-dev  && apt-get clean

ENV PYENV_ROOT /opt/.pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /opt/.pyenv/shims:/opt/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN pyenv install 3.8.10
RUN pyenv global 3.8.10

RUN pip  install -U pip

# thanks to libatlas-base-dev (base! not libatlas-dev), it will link to atlas
COPY requirements.txt requirements.txt
RUN python -m pip install cython numpy&&pip3 install pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html \
&&python -m pip install -r requirements.txt&&pip3 cache purge 



RUN python -m nltk.downloader popular && \ 
    python -m spacy download en_core_web_sm && \ 
    python -m spacy download xx_ent_wiki_sm

RUN pyenv rehash

RUN jupyter contrib nbextension install --system && \
    jupyter nbextensions_configurator enable --system && \
    jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
    jupyter labextension install @jupyterlab/toc && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager

# RUN ln -s /usr/local/cuda-11.2 /usr/local/nvidia && \
#     ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so.10 && \
#     ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.10


RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
        dpkg-reconfigure --frontend=noninteractive locales

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
VOLUME ["/notebook", "/jupyter/certs"]
WORKDIR /notebook

ADD test_scripts /test_scripts
ADD jupyter /jupyter
COPY entrypoint.sh /entrypoint.sh
COPY hashpwd.py /hashpwd.py

ENV JUPYTER_CONFIG_DIR="/jupyter"
RUN pip install youtokentome ipymarkup seqeval livelossplot && pip cache purge 
ENTRYPOINT ["/entrypoint.sh"]
CMD [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--port=8888"]
EXPOSE 8888
