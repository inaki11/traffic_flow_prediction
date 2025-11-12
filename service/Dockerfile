FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG CONDA_DIR=/opt/conda
ARG CONDA_ENV_NAME=sacyr
ARG PYTHON_VER=3.11
ENV PATH=${CONDA_DIR}/bin:${PATH}
WORKDIR /app

# 1) Paquetes base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget ca-certificates git bzip2 build-essential curl libgl1 cron dos2unix \
    && rm -rf /var/lib/apt/lists/*

# 2) Instalar Miniforge (conda-forge por defecto, sin TOS)
RUN wget --no-verbose https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

# 3) Configuración de conda
RUN ${CONDA_DIR}/bin/conda config --set always_yes yes --set changeps1 no --set ssl_verify no && \
    ${CONDA_DIR}/bin/conda update -n base conda

# 4) Copiar requirements.txt primero (aprovechar cache)
COPY requirements.txt /app/requirements.txt

# 5) Crear entorno conda + instalar PyTorch CUDA 11.8 + requirements
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VER} pip -y && \
    /bin/bash -lc "source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    conda install -n ${CONDA_ENV_NAME} -c conda-forge -c nvidia -c pytorch pytorch torchvision torchaudio pytorch-cuda=11.8 -y && \
    pip install -r /app/requirements.txt" && \
    conda clean -afy

# 6) Copiar aplicación
COPY main.py /app/main.py
RUN dos2unix /app/main.py
COPY simulacion_TR.py /app/simulacion_TR.py
RUN dos2unix /app/simulacion_TR.py
COPY a3_22_24_filtered.csv /app/a3_22_24_filtered.csv
RUN dos2unix /app/a3_22_24_filtered.csv
COPY /models /app/models


# 7) Configurar crontab
COPY crontab.txt /etc/cron.d/crontab
RUN chmod 0644 /etc/cron.d/crontab && dos2unix /etc/cron.d/crontab

# 8) Script de inicio
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
RUN dos2unix /app/start.sh

ENTRYPOINT ["/app/start.sh"]

