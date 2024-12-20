ARG base_image
FROM ${base_image}

ARG SPARK_VER=3.5.3
ARG SYNAPSEML_VER=1.0.8
ARG SLAMA_VER=0.4.1
ARG LIGHTGBM_VER=3.2.1
ARG spark_jars_cache=jars_cache

USER root

RUN pip install pyspark==${SPARK_VER}

#USER ${spark_id}
RUN mkdir -p /src

COPY docker/spark-lama/download_jars.py /src/download_jars.py

RUN python3 /src/download_jars.py

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt
# RUN pip install torchvision==0.9.1

COPY dist/sparklightautoml-${SLAMA_VER}-py3-none-any.whl /tmp/sparklightautoml-${SLAMA_VER}-py3-none-any.whl
RUN pip install /tmp/sparklightautoml-${SLAMA_VER}-py3-none-any.whl

RUN pip install --upgrade lightgbm==${LIGHTGBM_VER}

COPY examples/spark /examples

COPY jars /root/jars
COPY examples /examples

ENV PYSPARK_PYTHON=python3

WORKDIR /root
