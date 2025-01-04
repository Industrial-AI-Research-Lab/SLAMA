from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import log_exec_time


def main():
    persistence_manager = PlainCachePersistenceManager()

    dataset_name = "used_cars_dataset"
    # dataset_name = "lama_test_dataset"
    # dataset_name = "company_bankruptcy_dataset"
    load_dataset_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama"
    save_dataset_path = f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/simple_{dataset_name}_1part.slama"

    with log_exec_time():
        sdataset = SparkDataset.load(
            path=load_dataset_path,
            persistence_manager=persistence_manager,
            partitions_num=1
        )

        sdataset.save(save_dataset_path)


if __name__ == "__main__":
    main()
