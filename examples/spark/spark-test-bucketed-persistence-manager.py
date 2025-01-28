from examples.spark.examples_utils import get_spark_session


def main():
    spark = get_spark_session()

    df = spark.read.parquet("hdfs://node21.bdcl:9000/opt/spark_data/lama_test_dataset.csv")

    id_col = "SK_ID_CURR"

    (
        df.repartition(6, id_col)
        .write.mode("overwrite")
        .bucketBy(6, id_col)
        .sortBy(id_col)
        .saveAsTable("test", format="parquet")
    )

    ds = spark.table("test")

    print(f"DATA in TABLE: {ds.count()}")


if __name__ == "__main__":
    main()
