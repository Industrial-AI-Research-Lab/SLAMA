from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType, NumericType

from examples_utils import get_spark_session
from pyspark.sql import functions as sf


def main():
    spark = get_spark_session()

    df = (
        spark.read.format("parquet")
        .option("header", True)
        .option("inferSchema", True)
        .load(
            "hdfs://node21.bdcl:9000/tmp/bad_dataset_large.parquet"
        )
    )
    # print dataset size
    print("records read: " + str(df.count()))
    print("Schema: ")
    df.printSchema()

    # df = df.na.fill(0.0)
    df = df.repartition(2).cache()
    # df = df.select('*', sf.explode(sf.lit(list(range(100)))).alias("_tmp")).drop("_tmp")
    # df.write.parquet("hdfs://node21.bdcl:9000/tmp/bad_dataset_large.parquet")
    # return
    df.count()

    feature_cols = [c for c in df.columns if c not in ['TARGET', 'is_val']]

    row = df.select(
        sf.count("*").alias("count"),
        *[
            sf.mean((sf.isnull(feature) | sf.isnan(feature)).astype(IntegerType())).alias(f"{feature}")
            for feature in feature_cols
            if isinstance(df.schema[feature].dataType, NumericType)
        ],
        *[
            sf.mean((sf.isnull(feature)).astype(IntegerType())).alias(f"{feature}")
            for feature in feature_cols
            if not isinstance(df.schema[feature].dataType, NumericType)
        ],
    ).first()

    features = {
        col: rate for col, rate in row.asDict().items()
        if rate < 0.00000000000001 and col not in ['TARGET', 'is_val', 'reader_fold_num']
    }

    feature_cols = sorted(features.keys())
    # feature_cols = feature_cols[:10] + feature_cols[:20]
    # feature_cols = feature_cols[:25]

    # df = df.na.fill(0.0)
    # df = df.cache()
    # df.count()

    # feature_cols = ["AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL"]

    train, test = df.randomSplit([0.85, 0.15], seed=1)

    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["TARGET", "features"]
    test_data = featurizer.transform(test)["TARGET", "features"]

    from synapse.ml.lightgbm import LightGBMClassifier

    model = LightGBMClassifier(
        objective="binary", featuresCol="features", labelCol="TARGET", isUnbalance=True, executionMode="streaming"
    )

    model = model.fit(train_data)

    predictions = model.transform(test_data)
    pdf = predictions.limit(10).toPandas()

    print(pdf.iloc[:10])

    print("I'M HERE")


if __name__ == "__main__":
    main()
