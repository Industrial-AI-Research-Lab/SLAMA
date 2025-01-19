from tqdm import tqdm

from pyspark.sql import functions as sf
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import log_exec_time


def main():
    persistence_manager = PlainCachePersistenceManager()

    datasets = [
        # "used_cars_dataset",
        # "lama_test_dataset",
        # "company_bankruptcy_dataset",
        "adv_used_cars_dataset"
    ]

    for dataset_name in tqdm(datasets, desc="Processing datasets"):
        load_dataset_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama"
        save_dataset_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}_1part.slama"

        with log_exec_time():
            sdataset = SparkDataset.load(
                path=load_dataset_path,
                persistence_manager=persistence_manager,
                partitions_num=1
            )

            sdataset.save(save_dataset_path)


def main_convert_to_csv():
    persistence_manager = PlainCachePersistenceManager()

    datasets = [
        # "used_cars_dataset",
        # "lama_test_dataset",
        # "company_bankruptcy_dataset",
        # "adv_used_cars_dataset",
        "adv_small_used_cars_dataset"
    ]

    for dataset_name in tqdm(datasets, desc="Processing datasets"):
        load_dataset_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama"
        save_dataset_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/CSV/{dataset_name}.csv"

        with log_exec_time():
            sdataset = SparkDataset.load(
                path=load_dataset_path,
                persistence_manager=persistence_manager,
                partitions_num=1
            )
            df = sdataset.data
            df = df.na.fill(0.0)
            df = df.select(
                *(
                    sf.col(c).alias(c.replace('[', '(').replace(']', ')'))
                    for c in df.columns
                )
            )

            df.write.csv(save_dataset_path, header=True, encoding="UTF-8")


if __name__ == "__main__":
    main_convert_to_csv()
