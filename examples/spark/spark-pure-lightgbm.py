from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType, NumericType
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from examples_utils import get_spark_session

from pyspark.sql import functions as sf


def main():
    spark = get_spark_session()

    run_params = {
        'featuresCol': 'Mod_0_LightGBM_vassembler_features',
        # 'labelCol': 'TARGET',
        'labelCol': "price",
        # 'validationIndicatorCol': 'is_val',
        'verbosity': 1,
        # "dataTransferMode": "bulk",
        # "dataTransferMode": "streaming",
        "executionMode": "streaming",
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
        # 'objective': 'binary',
        # 'metric': 'auc',
        'objective': 'regression',
        'metric': 'rmse',
        'numTasks': None,
        'numThreads': None,
        'maxStreamingOMPThreads': 1,
        # 'rawPredictionCol': 'raw_prediction',
        # 'probabilityCol': 'Mod_0_LightGBM_prediction_0',
        'predictionCol': 'prediction',
        # 'isUnbalance': True
    }

    # train_df = spark.read.parquet("hdfs://node21.bdcl:9000/tmp/bad_dataset.parquet")
    # train_df = train_df.repartition(4).cache()
    # train_df.fillna(0.0).sample(0.01).repartition(1).write.csv("hdfs://node21.bdcl:9000/tmp/bd.csv", header=True, mode="overwrite")

    # train_df = spark.read.csv("hdfs://node21.bdcl:9000/tmp/bd.csv", header=True, inferSchema=True)
    # train_df = train_df.repartition(4).cache()
    # size = train_df.count()
    #
    # print(f"DATASET SIZE: {size}")
    #
    # # features = [c for c in train_df.columns if c != run_params['labelCol']]
    # features = [
    #     "ord__back_legroom",
    #     "ord__bed",
    #     "ord__bed_height",
    #     "ord__bed_length",
    #     "ord__body_type",
    #     "ord__cabin",
    #     "ord__city",
    #     "ord__city_fuel_economy",
    #     "ord__engine_cylinders",
    #     "ord__engine_displacement",
    #     "ord__engine_type",
    #     "ord__fleet",
    #     "ord__frame_damaged",
    #     "ord__franchise_dealer",
    #     "ord__franchise_make",
    #     "ord__front_legroom",
    #     "ord__fuel_tank_volume",
    #     "ord__fuel_type",
    #     "ord__has_accidents",
    #     "ord__height",
    #     "ord__highway_fuel_economy",
    #     "ord__horsepower",
    #     "ord__isCab",
    #     "ord__is_cpo",
    #     "ord__is_new",
    #     "ord__is_oemcpo",
    #     "ord__length",
    #     "ord__listing_color",
    #     "ord__make_name",
    #     "ord__maximum_seating",
    #     "ord__model_name",
    #     "ord__owner_count",
    #     "ord__power",
    #     "ord__salvage",
    #     "ord__seller_rating",
    #     "ord__theft_title",
    #     "ord__torque",
    #     "ord__transmission",
    #     "ord__transmission_display",
    #     "ord__vin",
    #     "ord__wheel_system",
    #     "ord__wheel_system_display",
    #     "ord__wheelbase",
    #     "ord__width"
    # ]
    #
    # # # doesn't work
    # # features = features[:5]
    # # # works
    # # features = features[:1]
    # # # doesn't work
    # # features = features[:3]
    # # # works
    # # features = features[2:3]
    # # # doesn't work
    # # features = features[1:3]
    # # # works
    # # features = features[1:2]
    #
    # features = features[1:3]
    # features = ["ord__bed", "ord__bed_height"]
    # train_df = train_df.select("price", sf.col("ord__bed").astype("double"), sf.col("ord__bed_height").astype("double"))

    # suprisignly, it works
    train_df = spark.read.csv("hdfs://node21.bdcl:9000/tmp/bd_2cols.csv", header=True, inferSchema=True)
    features = ["ord__bed", "ord__bed_height"]

    # features = ["latitude", "longitude"]

    # features = [
    #     "daysonmarket",
    #     "latitude",
    #     "listing_id",
    #     "longitude",
    #     "mileage",
    #     "savings_amount"
    # ]

    # train_df = train_df.na.fill(0)
    #
    # row = train_df.select(
    #     sf.count("*").alias("count"),
    #     *[
    #         sf.mean((sf.isnull(feature) | sf.isnan(feature)).astype(IntegerType())).alias(f"{feature}_nan_rate")
    #         for feature in features
    #         if isinstance(train_df.schema[feature].dataType, NumericType)
    #     ],
    #     *[
    #         sf.mean((sf.isnull(feature)).astype(IntegerType())).alias(f"{feature}_nan_rate")
    #         for feature in features
    #         if not isinstance(train_df.schema[feature].dataType, NumericType)
    #     ],
    # ).first()

    assembler = VectorAssembler(
        inputCols=features, outputCol=run_params['featuresCol'], handleInvalid="skip"
    )

    df = assembler.transform(train_df)

    print(f"ASSEMBLED DATASET SIZE: {df.count()}")

    # lgbm = LightGBMClassifier(**run_params)
    lgbm = LightGBMRegressor(**run_params)

    ml_model = lgbm.fit(df)


if __name__ == "__main__":
    main()
