FROM bitnami/spark:3.4.1  

USER root

# Установка Python зависимостей
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    delta-spark==2.4.0 \
    pyspark==3.4.1 \
    pandas \
    py4j==0.10.9.7 \
    scikit-learn

# Создаем структуру папок
RUN mkdir -p /lab3/data/bronze \
    /lab3/data/silver \
    /lab3/data/gold \
    /lab3/logs \
    /lab3/mlruns

WORKDIR /lab3

# Копируем файлы
COPY ./src/spark.py /lab3/src/
COPY ./src /lab3/src
COPY ./data/Online_Retail_Sample_100k.csv /lab3/data/Online_Retail_Sample_100k.csv

COPY ./src/run.sh /lab3/run.sh
RUN chmod +x /lab3/run.sh

ENV SPARK_LOCAL_DIRS=/tmp/spark
ENV SPARK_WORKER_MEMORY=1g

ENTRYPOINT ["/lab3/run.sh"]