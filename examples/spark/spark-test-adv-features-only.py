import logging.config

from examples_utils import get_dataset
from examples_utils import get_spark_session
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import log_exec_time
from sparklightautoml.utils import logging_config

logging.config.dictConfig(logging_config(log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 5
    ml_alg_kwargs = {
        "auto_unique_co": 10,
        "max_intersection_depth": 1,
        "multiclass_te_co": 3,
        "output_categories": False,
        "debug_only_le_without_te": False,
        "top_intersections": 4,
    }
    # dataset_name = "lama_test_dataset"
    # dataset_name = "small_used_cars_dataset"
    # dataset_name = "used_cars_dataset"
    # dataset_name = "used_cars_dataset_4x"
    dataset_name = "used_cars_dataset_10x"
    # dataset_name = "used_cars_dataset_40x"
    # dataset_name = "company_bankruptcy_dataset"
    # dataset_name = "company_bankruptcy_dataset_100x"
    # dataset_name = "company_bankruptcy_dataset_10000x"
    dataset = get_dataset(dataset_name)

    # TODO: there is some problem with composite persistence manager on kubernetes. Need to research later.
    # persistence_manager = get_persistence_manager()
    persistence_manager = PlainCachePersistenceManager()

    with log_exec_time():
        train_df = dataset.load()

        # optional part
        execs = int(spark.conf.get("spark.executor.instances", "1"))
        cores = int(spark.conf.get("spark.executor.cores", "8"))
        train_df = train_df.repartition(execs * cores * 1)

        task = SparkTask(dataset.task_type)

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False, )
        spipe = SparkLGBAdvancedPipeline(**ml_alg_kwargs)

        sdataset = sreader.fit_read(train_df, roles=dataset.roles, persistence_manager=persistence_manager)
        sdataset = spipe.fit_transform(sdataset)
        name_prefix = "half_adv" if ml_alg_kwargs.get("debug_only_le_without_te", False) else "adv"
        # sdataset.save(
        #     f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{name_prefix}_{dataset_name}.slama",
        #     save_mode="overwrite",
        #     num_partitions=1
        # )

        # # How to load
        # sdataset = SparkDataset.load(
        #     path=f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama",
        #     persistence_manager=persistence_manager
        # )
        #
        # print("Dataset Features: ")
        # pprint(sdataset.features)
        # size = sdataset.data.count()
        # print(f"Dataset size: {size}")

    logger.info("Finished")

    spark.stop()
