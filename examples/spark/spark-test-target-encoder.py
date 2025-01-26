import os
import sys
from typing import Optional

from lightautoml.pipelines.utils import get_columns_by_role
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.transformers.categorical import SparkTargetEncoderEstimator


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


def get_lightgbm_params(dataset_name: str) -> str:
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

    return data_path


def load_data(spark: SparkSession, data_path: str, partitions_coefficient: int = 1) -> SparkDataset:
    execs = int(spark.conf.get("spark.executor.instances", "1"))
    cores = int(spark.conf.get("spark.executor.cores", "8"))
    num_partitions = execs * cores * partitions_coefficient

    sdataset = SparkDataset.load(
        path=data_path,
        persistence_manager=PlainCachePersistenceManager(),
        partitions_num=num_partitions
    )

    sdataset = sdataset.persist()

    return sdataset


def main():
    dataset_name = sys.argv[1]

    print(f"Working with dataset: {dataset_name}")

    data_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama/data.parquet"

    spark = get_spark_session()

    dataset = load_data(spark=spark, data_path=data_path)

    # prepare estimator
    feats_to_select = []
    for i in ["auto", "oof", "int", "ohe"]:
        feats = get_columns_by_role(dataset, "Category", encoding_type=i)
        feats_to_select.extend(feats)

    roles = {f: dataset.roles[f] for f in feats_to_select}

    estimator = SparkTargetEncoderEstimator(
        input_cols=feats_to_select,
        input_roles=roles,
        task_name=dataset.task.name,
        target_column=dataset.target_column,
        folds_column=dataset.folds_column,
    )

    # fit
    transformer = estimator.fit(dataset.data)

    # save
    transformer = PipelineModel(stages=[transformer])
    transformer.write().overwrite().save(
        f"hdfs://node21.bdcl:9000/opt/transformers/target_encoder_{dataset_name}.transformer"
    )

    # process
    df = transformer.transform(dataset.data)
    df.count()


if __name__ == "__main__":
    main()
