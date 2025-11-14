# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Dockerfile: eplus-sinergym-py3.12                                        │
# └───────────────────────────────────────────────────────────────────────────┘

# 1) Base on official Python 3.12 slim (Debian-based)
FROM python:3.12-slim

# avoid prompts
ARG DEBIAN_FRONTEND=noninteractive

# 2) Install system deps
RUN apt-get update \ 
&& apt-get install -y --no-install-recommends \
wget \
tar \
libx11-6 \
libexpat1 \
libgomp1 \
&& rm -rf /var/lib/apt/lists/*

# Download and install EnergyPlus
ARG EPLUS_VERSION=24.1.0
ARG EPLUS_URL=https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64.tar.gz

RUN wget ${EPLUS_URL} -O energyplus.tar.gz \
    && mkdir -p /opt/EnergyPlus \
    && tar -xzf energyplus.tar.gz --strip-components=1 -C /opt/EnergyPlus \
    && rm energyplus.tar.gz

# Add EnergyPlus Python modules to PYTHONPATH
#RUN echo "export PYTHONPATH=/opt/EnergyPlus:$PYTHONPATH" >> ~/.bashrc
ENV PYTHONPATH=/opt/EnergyPlus:${PYTHONPATH}
ENV EPLUS_PATH=/opt/EnergyPlus
# 4) Copy and install your Python dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
&& pip install --no-cache-dir -r requirements.txt

# 5) Copy & install your custom sinergym-a403 package
COPY sinergym-a403 /app/sinergym
# ensure there's a build file—if you have a pyproject.toml in sinergym-a403, pip will use it
RUN pip install --no-cache-dir /app/sinergym
# Execute the command
CMD ["/bin/bash"]

