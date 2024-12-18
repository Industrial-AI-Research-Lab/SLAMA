from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType, NumericType
from synapse.ml.lightgbm import LightGBMClassifier

from examples_utils import get_spark_session

from pyspark.sql import functions as sf


def main():
    spark = get_spark_session()

    run_params = {
        'featuresCol': 'Mod_0_LightGBM_vassembler_features',
        'labelCol': 'TARGET',
        'validationIndicatorCol': 'is_val',
        'verbosity': 1,
        "dataTransferMode": "bulk",
        # 'executionMode': 'bulk',
        'useSingleDatasetMode': True,
        'useBarrierExecutionMode': False,
        'isProvideTrainingMetric': True,
        'chunkSize': 4000000,
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
        'numIterations': 2000,
        'earlyStoppingRound': 200,
        'objective': 'binary',
        'metric': 'auc',
        'numTasks': None,
        'numThreads': None,
        'rawPredictionCol': 'raw_prediction',
        'probabilityCol': 'Mod_0_LightGBM_prediction_0',
        'predictionCol': 'prediction',
        'isUnbalance': True
    }

    train_df = spark.read.parquet("hdfs://node21.bdcl:9000/tmp/bad_dataset.parquet")
    train_df = train_df.cache()
    size = train_df.count()

    print(f"DATASET SIZE: {size}")

    features = [c for c in train_df.columns if c != run_params['labelCol']]

    train_df = train_df.na.fill(0)

    row = train_df.select(
        sf.count("*").alias("count"),
        *[
            sf.mean((sf.isnull(feature) | sf.isnan(feature)).astype(IntegerType())).alias(f"{feature}_nan_rate")
            for feature in features
            if isinstance(train_df.schema[feature].dataType, NumericType)
        ],
        *[
            sf.mean((sf.isnull(feature)).astype(IntegerType())).alias(f"{feature}_nan_rate")
            for feature in features
            if not isinstance(train_df.schema[feature].dataType, NumericType)
        ],
    ).first()

    assembler = VectorAssembler(
        inputCols=features, outputCol=run_params['featuresCol'], handleInvalid="error"
    )

    df = assembler.transform(train_df)

    print(f"ASSEMBLED DATASET SIZE: {df.count()}")

    lgbm = LightGBMClassifier(**run_params)

    ml_model = lgbm.fit(df)


if __name__ == "__main__":
    main()
