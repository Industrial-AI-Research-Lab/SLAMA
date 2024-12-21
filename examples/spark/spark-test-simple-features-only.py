import logging.config
from pprint import pprint

from examples_utils import get_dataset
from examples_utils import get_persistence_manager
from examples_utils import get_spark_session
from examples_utils import prepare_test_and_train
from examples_utils import check_columns
from pyspark.ml import PipelineModel
from pyspark.sql import functions as sf

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import log_exec_time
from sparklightautoml.utils import logging_config
from sparklightautoml.validation.iterators import SparkFoldsIterator


logging.config.dictConfig(logging_config(log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 2
    dataset_name = "lama_test_dataset"
    # dataset_name = "used_cars_dataset"
    dataset = get_dataset(dataset_name)

    # TODO: there is some problem with composite persistence manager on kubernetes. Need to research later.
    # persistence_manager = get_persistence_manager()
    persistence_manager = PlainCachePersistenceManager()

    with log_exec_time():
        train_df, test_df = prepare_test_and_train(dataset, seed)

        print(f"TRAIN_DF size: {train_df.count()}")
        print(f"TEST_DF size: {test_df.count()}")

        task = SparkTask(dataset.task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(train_df, roles=dataset.roles, persistence_manager=persistence_manager)

        sdataset = SparkLGBSimpleFeatures().fit_transform(sdataset)

        sdataset.save(f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama")

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
