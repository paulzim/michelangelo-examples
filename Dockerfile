# syntax=docker/dockerfile:1
###############################################################################
# $ docker build -t <IMAGE_NAME> .
###############################################################################

# Use a base image with Python installed
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Install all system dependencies in a single layer.
# --mount=type=cache persists the apt cache across builds so packages are not
# re-downloaded on every run. sharing=locked prevents cache corruption when
# two platform builds run concurrently on the same runner.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    software-properties-common \
    vim \
    wget \
    build-essential \
    libssl-dev \
    libreadline-dev \
    libbz2-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    zlib1g-dev \
    openjdk-11-jdk

# Set Python version
ENV PYTHON_VERSION=3.10.14

# Download and install Python from source
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xvf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-$PYTHON_VERSION*

# Ensure python3.10 is the default
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python

# Install pip and uv
RUN python --version
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv --version


# --------------------------
# 3. Install Spark (manual)
# --------------------------
ENV SPARK_VERSION=3.5.5
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

RUN curl -L https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  | tar -xz -C /opt && \
  ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME}

ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

# -----------------------------
# 4. Add Hadoop AWS & AWS SDK
# -----------------------------
# Create JAR directory if needed
RUN mkdir -p ${SPARK_HOME}/jars

# Download hadoop-aws and aws-java-sdk-bundle JARs for S3 support
RUN curl -L -o ${SPARK_HOME}/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -L -o ${SPARK_HOME}/jars/aws-java-sdk-bundle-1.11.1026.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar

RUN mkdir -p ${SPARK_HOME}/conf && \
    echo "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> ${SPARK_HOME}/conf/spark-defaults.conf && \
    echo "spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain" >> ${SPARK_HOME}/conf/spark-defaults.conf


###
### Install Java
###

# Detect JAVA_HOME at build time so the image works on both arm64 and amd64.
RUN JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME=${JAVA_HOME}" >> /etc/environment && \
    echo "export JAVA_HOME=${JAVA_HOME}" >> /etc/profile.d/java.sh
ARG TARGETARCH
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-${TARGETARCH}
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

ENV PYTHONPATH=/app

# Which project's dependencies to install -- one image per project (use
# case), shared by all of that project's pipelines, matching
# michelangelo-examples' per-project extras (not one monolithic bundle the
# way core michelangelo's examples:main image installs everything at once).
ARG PROJECT_EXTRA=california-housing

# Copy the application code
COPY . .

# Install python dependencies via uv, only this project's extra.
RUN uv venv /app/.venv
RUN uv pip install --python /app/.venv/bin/python ".[${PROJECT_EXTRA}]"

# Activate the venv by default
ENV PATH="/app/.venv/bin:$PATH"

# michelangelo's own sandbox SparkApplication template hardcodes
# `local:///app/michelangelo/uniflow/core/run_task.py`, assuming the package
# sits directly under /app rather than in a nested venv's site-packages.
# TODO(upstream, https://github.com/michelangelo-ai/michelangelo/issues/1516):
# fix michelangelo's spark-app.yaml to discover the installed package location
# dynamically (or use a `python -m` entrypoint) instead of hardcoding this
# path; once fixed, this symlink can be removed.
RUN ln -s "$(/app/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/michelangelo" /app/michelangelo

# spark-submit's PythonRunner spawns the driver/executor Python subprocess via
# PYSPARK_PYTHON/PYSPARK_DRIVER_PYTHON, not the container's PATH — without
# these, it falls back to the system python3 (no project deps installed).
ENV PYSPARK_PYTHON=/app/.venv/bin/python
ENV PYSPARK_DRIVER_PYTHON=/app/.venv/bin/python

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
