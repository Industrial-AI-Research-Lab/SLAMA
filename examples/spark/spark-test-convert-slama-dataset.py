from tqdm import tqdm

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


if __name__ == "__main__":
    main()
