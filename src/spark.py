import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, to_timestamp, dayofweek, month
from pyspark.sql.functions import concat, lit, expr
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable

# Конфигурации Spark
def configure_spark_session():
    builder = SparkSession.builder \
        .appName("RetailAnalyticsPipeline") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g")

    return configure_spark_with_delta_pip(builder).getOrCreate()

# Инициализация Spark
spark = configure_spark_session()

# Настройка MLflow
mlflow.set_tracking_uri("file:/lab3/logs/mlruns")
mlflow.set_experiment("/Retail/GBT_OrderValuePrediction")

def load_initial_data(file_path: str) -> DataFrame:
    schema = StructType([
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True),
        StructField("UnitPrice", FloatType(), True),
        StructField("CustomerID", StringType(), True),
        StructField("Country", StringType(), True)
    ])

    df = spark.read \
        .schema(schema) \
        .option("header", "true") \
        .csv(file_path)

    df = df.withColumn(
        "InvoiceDate",
        to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss")
    ).withColumn(
        "TotalAmount", col("Quantity") * col("UnitPrice")
    ).filter((col("Quantity") > 0) & (col("UnitPrice") > 0))

    return df.repartition(1)

def process_data(df: DataFrame) -> DataFrame:
    processed = df \
        .withColumn("DayOfWeek", dayofweek(col("InvoiceDate"))) \
        .withColumn("Month", month(col("InvoiceDate"))) \
        .withColumn("HourOfDay", (col("InvoiceDate").cast("long") % 86400) / 3600) \
        .withColumn("CustomerID", when(col("CustomerID").isNull(), "Unknown").otherwise(col("CustomerID"))) \
        .withColumn("Description", when(col("Description").isNull(), "Unknown").otherwise(col("Description"))) \
        .withColumn("NegativeQuantity", when(col("Quantity") < 0, 1).otherwise(0))
    
    return processed

def apply_target_encoding(train: DataFrame, test: DataFrame, cols: list) -> tuple:
    global_mean = train.select(mean("TotalAmount")).first()[0]
    
    for col_name in cols:
        target_mean = train.groupBy(col_name) \
                          .agg(mean("TotalAmount").alias(f"{col_name}_encoded"))
        
        train = train.join(target_mean, on=col_name, how="left") \
                    .fillna({f"{col_name}_encoded": global_mean})
        
        test = test.join(target_mean, on=col_name, how="left") \
                  .fillna({f"{col_name}_encoded": global_mean})
    
    return train, test

def save_data(train: DataFrame, test: DataFrame, path: str):
    train.write.format("delta").mode("overwrite").save(f"{path}/train")
    test.write.format("delta").mode("overwrite").save(f"{path}/test")

def optimize_delta_table(path: str):
    delta_table = DeltaTable.forPath(spark, path)
    delta_table.optimize().executeZOrderBy(["Country"])

def prepare_final_dataset(df: DataFrame, features: list, target: str) -> DataFrame:
    return df.select(features + [target])

def train_model(train_data: DataFrame, test_data: DataFrame, features: list, target_col: str):
    try:
        with mlflow.start_run(run_name="GBT_OrderValuePrediction"):
            assembler = VectorAssembler(
                inputCols=features,
                outputCol="features",
                handleInvalid="skip"
            )
            
            gbt = GBTRegressor(
                labelCol=target_col,
                featuresCol="features",
                maxDepth=5,
                maxIter=100,
                stepSize=0.1,
                maxBins=32,
                seed=42
            )
            
            pipeline = Pipeline(stages=[assembler, gbt])
            model = pipeline.fit(train_data)
            
            predictions = model.transform(test_data)
            
            evaluator = RegressionEvaluator(labelCol=target_col)
            metrics = {
                "rmse": evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}),
                "mae": evaluator.evaluate(predictions, {evaluator.metricName: "mae"}),
                "r2": evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            }
            
            mlflow.log_metrics(metrics)
            mlflow.spark.log_model(model, "model")
            
            print(f"Model metrics: {metrics}")
            return model
            
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        spark.stop()
        raise

# --- Основной процесс ---
with mlflow.start_run(run_name="DataIngestion"):
    initial_data = load_initial_data("/lab3/data/retail.csv")
    initial_data.write.format("delta").mode("overwrite").save("/lab3/data/bronze/retail_data")

with mlflow.start_run(run_name="DataProcessing"):
    bronze_data = spark.read.format("delta").load("/lab3/data/bronze/retail_data")
    processed_data = process_data(bronze_data)
    processed_data.cache().count()
    
    train_raw, test_raw = processed_data.randomSplit([0.8, 0.2], seed=42)
    
    train_encoded, test_encoded = apply_target_encoding(
        train_raw, test_raw, ["Country", "StockCode"]
    )
    
    save_data(train_encoded, test_encoded, "/lab3/data/silver")
    optimize_delta_table("/lab3/data/silver/train")
    optimize_delta_table("/lab3/data/silver/test")

with mlflow.start_run(run_name="FeatureEngineering"):
    train_data = spark.read.format("delta").load("/lab3/data/silver/train")
    test_data = spark.read.format("delta").load("/lab3/data/silver/test")
    
    feature_columns = [
        "StockCode_encoded", "Country_encoded",
        "DayOfWeek", "Month", "HourOfDay",
        "UnitPrice", "Quantity", "NegativeQuantity"
    ]
    
    final_train = prepare_final_dataset(train_data, feature_columns, "TotalAmount")
    final_test = prepare_final_dataset(test_data, feature_columns, "TotalAmount")
    
    final_train.write.format("delta").mode("overwrite").save("/lab3/data/gold/train")
    final_test.write.format("delta").mode("overwrite").save("/lab3/data/gold/test")

# Обучение модели
final_train = spark.read.format("delta").load("/lab3/data/gold/train")
final_test = spark.read.format("delta").load("/lab3/data/gold/test")

trained_model = train_model(final_train, final_test, feature_columns, "TotalAmount")