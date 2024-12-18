from pyspark.ml.feature import VectorAssembler

from examples_utils import get_spark_session


def main():
    spark = get_spark_session()

    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(
            "hdfs://node21.bdcl:9000/opt/spark_data/company_bankruptcy_prediction_data.csv"
        )
    )
    # print dataset size
    print("records read: " + str(df.count()))
    print("Schema: ")
    df.printSchema()

    train, test = df.randomSplit([0.85, 0.15], seed=1)

    feature_cols = df.columns[1:]
    featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = featurizer.transform(train)["Bankrupt?", "features"]
    test_data = featurizer.transform(test)["Bankrupt?", "features"]

    from synapse.ml.lightgbm import LightGBMClassifier

    model = LightGBMClassifier(
        objective="binary", featuresCol="features", labelCol="Bankrupt?", isUnbalance=True
    )

    model = model.fit(train_data)

    predictions = model.transform(test_data)
    pdf = predictions.limit(10).toPandas()

    print(pdf.iloc[:10])

    print("I'M HERE")


if __name__ == "__main__":
    main()
