FROM jenkins/jenkins:lts-jdk21

USER root

# Install Python 3, pip, venv, git, curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Prevent PEP 668 externally-managed-environment errors in this container
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Safe directory configuration for mounted Git repositories
RUN git config --system --add safe.directory '*'

# Pre-install Lab 2 ML dependencies to ensure fast and reliable pipeline execution
RUN pip3 install --no-cache-dir pandas scikit-learn numpy

USER jenkins
