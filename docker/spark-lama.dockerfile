FROM python:3.9.9

RUN wget https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.25%2B9/OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz
RUN tar -xvf OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz
RUN mv jdk-11.0.25+9 /usr/local/lib/jdk-11
RUN ln -s /usr/local/lib/jdk-11/bin/java /usr/local/bin/java
RUN rm OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz

RUN mkdir -p /src
COPY dist/sparklightautoml_dev-0.3.2-py3-none-any.whl /src
RUN pip install /src/sparklightautoml_dev-0.3.2-py3-none-any.whl

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.1-spark3.3").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mkdir /src/jars
COPY jars/spark-lightautoml_2.12-0.1.1.jar /src/jars/

COPY examples /src/examples
COPY examples/data /opt/spark_data

WORKDIR /src

ENTRYPOINT ["python3"]
