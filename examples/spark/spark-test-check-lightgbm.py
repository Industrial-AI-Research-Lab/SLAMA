import logging
import os
import signal
import sys
import time
from typing import Optional, Any, Dict, Tuple
import psutil
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor


logger = logging.getLogger(__name__)


GENERAL_RUN_PARAMS = {
    'featuresCol': 'Mod_0_LightGBM_vassembler_features',
    'verbosity': 1,
    'useSingleDatasetMode': True,
    'useBarrierExecutionMode': False,
    'isProvideTrainingMetric': True,
    'chunkSize': 10_000,
    'defaultListenPort': 13614,
    'learningRate': 0.03,
    'numLeaves': 64,
    'featureFraction': 0.7,
    'baggingFraction': 0.7,
    'baggingFreq': 1,
    'maxDepth': -1,
    'minGainToSplit': 0.0,
    'maxBin': 255,
    'minDataInLeaf': 5,
    'numIterations': 50,
    'earlyStoppingRound': 200,
    # 'numTasks': None,
    'numThreads': 4,
    'matrixType': 'auto',
    'maxStreamingOMPThreads': 1,

    # 'dataTransferMode': 'bulk',
    # 'numTasks': 6,

    'dataTransferMode': 'streaming',
    # 'numTasks': 6
}


def get_lightgbm_params(spark: SparkSession, dataset_name: str) -> Tuple[str, Dict[str, Any]]:
    data_path = None
    match dataset_name:
        case "company_bankruptcy_dataset":
            dataset_specific_params = {
                'labelCol': "Bankrupt?",
                'objective': 'binary',
                'metric': 'auc',
                'rawPredictionCol': 'raw_prediction',
                'probabilityCol': 'Mod_0_LightGBM_prediction_0',
                'predictionCol': 'prediction',
                'isUnbalance': True
            }
        case "lama_test_dataset":
            dataset_specific_params = {
                'labelCol': 'TARGET',
                'objective': 'binary',
                'metric': 'auc',
                'rawPredictionCol': 'raw_prediction',
                'probabilityCol': 'Mod_0_LightGBM_prediction_0',
                'predictionCol': 'prediction',
                'isUnbalance': True
            }
        case "used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "small_used_cars_dataset":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/small_used_cars_dataset.slama/data.parquet"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "used_cars_dataset_10x":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/used_cars_dataset_10x.slama/data.parquet"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "used_cars_dataset_100x":
            data_path = "hdfs://node21.bdcl:9000/opt/preprocessed_datasets/used_cars_dataset_100x.slama/data.parquet"
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "adv_small_used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case "adv_used_cars_dataset":
            dataset_specific_params = {
                'labelCol': "price",
                'objective': 'regression',
                'metric': 'rmse',
                'predictionCol': 'prediction'
            }
        case _:
            raise ValueError("Unknown dataset")

    data_path = data_path or f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/CSV/{dataset_name}.csv"

    execs = int(spark.conf.get("spark.executor.instances", "1"))

    return data_path, {
        **GENERAL_RUN_PARAMS,
        **dataset_specific_params,
        "numTasks": execs
    }


def get_spark_session(partitions_num: Optional[int] = None):
    partitions_num = partitions_num if partitions_num else 6

    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
    else:

        extra_jvm_options = "-Dio.netty.tryReflectionSetAccessible=true "

        spark_sess = (
            SparkSession.builder.master(f"local[{partitions_num}]")
            # .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.8")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.8")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.1.jar")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.driver.extraJavaOptions", extra_jvm_options)
            .config("spark.executor.extraJavaOptions", extra_jvm_options)
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            .config("spark.cleaner.referenceTracking", "true")
            .config("spark.cleaner.periodicGC.interval", "1min")
            .config("spark.sql.shuffle.partitions", f"{partitions_num}")
            .config("spark.default.parallelism", f"{partitions_num}")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

    spark_sess.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    # spark_sess.sparkContext.setLogLevel("WARN")

    return spark_sess


def load_data(spark: SparkSession, data_path: str, partitions_coefficient: int = 1) -> DataFrame:
    if data_path.endswith('.csv'):
        data = spark.read.csv(data_path, header=True, inferSchema=True, encoding="UTF-8")
    else:
        data = spark.read.parquet(data_path)

    execs = int(spark.conf.get("spark.executor.instances", "1"))
    cores = int(spark.conf.get("spark.executor.cores", "8"))

    data = data.repartition(execs * cores * partitions_coefficient).cache()
    data.write.mode("overwrite").format("noop").save()

    return data


def load_test_and_train(
    spark: SparkSession, data_path: str, seed: int = 42, test_size: float = 0.2, partitions_coefficient: int = 1
) -> Tuple[DataFrame, DataFrame]:
    assert 0 <= test_size <= 1

    if data_path.endswith('.csv'):
        data = spark.read.csv(data_path, header=True, inferSchema=True, encoding="UTF-8")
    else:
        data = spark.read.parquet(data_path)

    # if "adv_small_used_cars_dataset" in data_path:
    #     data = data.select(
    #         *[
    #             c for c in data.columns if c in [
    #                 # 'engine_displacement',
    #                 # 'highway_fuel_economy',
    #                 # 'mileage',
    #                 # 'listing_id',
    #
    #                 'ord__bed_height',
    #                 'ord__is_oemcpo',
    #                 'ord__is_cpo',
    #
    #                 # 'daysonmarket',
    #                 # 'owner_count',
    #                 # 'horsepower',
    #                 # 'savings_amount',
    #                 # 'city_fuel_economy',
    #
    #                 # '_id', 'price',
    #                 # 'longitude',
    #                 # 'seller_rating',
    #                 # 'latitude'
    #             ]
    #         ],
    #         'price'
    #     )
    #     print("Removing bug-related columns from small_used_cars")

    # small adjustment in values making them non-categorial prevent SIGSEGV from happening
    # data = data.na.fill(0.0453)
    # data = data.select(
    #     *[
    #         (sf.col(c) + (sf.rand() / sf.lit(10.0)) + sf.lit(0.05)).alias(c)
    #         for c in data.columns if c not in ['_id', 'price']
    #     ],
    #     'price'
    # )

    execs = int(spark.conf.get("spark.executor.instances", "1"))
    cores = int(spark.conf.get("spark.executor.cores", "8"))

    data = data.repartition(execs * cores * partitions_coefficient).cache()
    data.write.mode("overwrite").format("noop").save()

    train_data, test_data = data.randomSplit([1 - test_size, test_size], seed)

    return train_data, test_data


def clean_java_processes():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        time.sleep(10)
        pids = [proc.pid for proc in psutil.process_iter() if "java" in proc.name()]
        print(f"Found unstopped java processes: {pids}")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except:
                logger.warning(f"Exception during killing the java process with pid {pid}", exc_info=True)


def main():
    dataset_name = sys.argv[1]

    print(f"Working with dataset: {dataset_name}")

    spark = get_spark_session()

    data_path, run_params = get_lightgbm_params(spark, dataset_name)

    train_df, test_df = load_test_and_train(spark=spark, data_path=data_path)

    print(f"ASSEMBLED DATASET SIZE: {train_df.count()}")

    features = [c for c in train_df.columns if c not in [run_params['labelCol'], '_id', 'reader_fold_num', 'is_val']]
    assembler = VectorAssembler(inputCols=features, outputCol=run_params['featuresCol'], handleInvalid="keep")

    match run_params['objective']:
        case 'regression':
            lgbm = LightGBMRegressor(**run_params)
        case 'binary':
            lgbm = LightGBMClassifier(**run_params)
        case _:
            raise ValueError()

    df = assembler.transform(train_df)
    model = lgbm.fit(df)
    print("Training is finished")

    df = assembler.transform(test_df)
    predicts_df = model.transform(df)
    predicts_df.write.mode("overwrite").format("noop").save()
    print("Predicting is finished")

    # time.sleep(600)

    spark.stop()
    clean_java_processes()


if __name__ == "__main__":
    main()
