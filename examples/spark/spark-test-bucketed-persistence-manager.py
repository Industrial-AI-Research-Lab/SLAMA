from examples_utils import get_spark_session


def main():
    spark = get_spark_session()

    df = spark.read.csv("hdfs://node21.bdcl:9000/opt/spark_data/sampled_app_train.csv", header=True, escape="")

    print(f"DATA: {df.count()}")

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
    k = 0


if __name__ == "__main__":
    main()
