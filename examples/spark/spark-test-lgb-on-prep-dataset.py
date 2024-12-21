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
        sdataset = SparkDataset.load(
            path=f"hdfs://node21.bdcl:9000/opt/preprocessed_datasets/{dataset_name}.slama",
            persistence_manager=persistence_manager
        )
        score = SparkTask(dataset.task_type).get_dataset_metric()

        iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()

        spark_ml_algo = SparkBoostLGBM(
            default_params={
              "numIterations": 50,
            },
            freeze_defaults=True,
            execution_mode="bulk",
        )

        ml_pipe = SparkMLPipeline(ml_algos=[spark_ml_algo])

        oof_preds_ds = ml_pipe.fit_predict(iterator)
        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

    logger.info("Finished")

    spark.stop()
