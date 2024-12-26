import logging.config

from examples_utils import get_dataset, load_data
from examples_utils import get_spark_session
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

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    persistence_manager = PlainCachePersistenceManager()

    seed = 42
    cv = 5
    # dataset_name = "lama_test_dataset"
    dataset_name = "small_used_cars_dataset_preproc"
    dataset = get_dataset(dataset_name)

    ml_alg_kwargs = {
        "auto_unique_co": 10,
        "max_intersection_depth": 3,
        "multiclass_te_co": 3,
        "output_categories": True,
        "top_intersections": 4,
    }

    with log_exec_time():
        data = load_data(dataset)

        task = SparkTask(dataset.task_type)
        score = task.get_dataset_metric()
        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        spark_ml_algo = SparkBoostLGBM(
            default_params={
                "numIterations": 50,
            },
            freeze_defaults=True,
            execution_mode="bulk"
        )
        spark_features_pipeline = SparkLGBSimpleFeatures()
        # spark_features_pipeline = SparkLGBAdvancedPipeline(**ml_alg_kwargs)

        ml_pipe = SparkMLPipeline(
            ml_algos=[spark_ml_algo],
            pre_selection=None,
            features_pipeline=spark_features_pipeline,
            post_selection=None,
        )

        sdataset = sreader.fit_read(data, roles=dataset.roles, persistence_manager=persistence_manager)

        iterator = SparkFoldsIterator(sdataset, n_folds=cv)

        oof_preds_ds = ml_pipe.fit_predict(iterator).persist()
        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

    logger.info("Finished")

    oof_preds_ds.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    persistence_manager.unpersist_all()

    spark.stop()
