FROM nvcr.io/nvidia/tritonserver:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/triton-benchmark

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        jq \
        numactl \
        openssh-server \
        procps \
        rsync \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -m 0755 /run/sshd

COPY requirements-pod.txt /tmp/requirements-pod.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-pod.txt \
    && rm /tmp/requirements-pod.txt

WORKDIR /opt/triton-benchmark
COPY triton_benchmark ./triton_benchmark
COPY scripts ./scripts
COPY docker/sshd_config /etc/ssh/sshd_config
COPY docker/start-pod.sh /usr/local/bin/start-pod
RUN chmod 0755 /usr/local/bin/start-pod

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/start-pod"]
