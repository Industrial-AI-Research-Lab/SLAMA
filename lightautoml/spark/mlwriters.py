import json
import logging
import os
import pickle
import time

from pathlib import Path
from typing import Union

from pyspark import SparkContext
from pyspark.ml.common import inherit_doc
from pyspark.ml.util import DefaultParamsReader
from pyspark.ml.util import MLReadable
from pyspark.ml.util import MLReader
from pyspark.ml.util import MLWritable
from pyspark.ml.util import MLWriter
from synapse.ml.lightgbm import LightGBMClassificationModel
from synapse.ml.lightgbm import LightGBMRegressionModel


logger = logging.getLogger(__name__)


class CommonPickleMLWritable(MLWritable):
    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Transformer instance."
        return СommonPickleMLWriter(self)


class CommonPickleMLReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return СommonPickleMLReader()


class СommonPickleMLWriter(MLWriter):
    """Implements saving an Estimator/Transformer instance to disk.
    Used when saving a trained pipeline.
    Implements MLWriter.saveImpl(path) method.
    """

    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        logger.info(f"Save {self.instance.__class__.__name__} to '{path}'")

        СommonPickleMLWriter.saveMetadata(self.instance, path, self.sc)

        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "transformer_class_instance.pickle"), 'wb') as handle:
            pickle.dump(self.instance, handle)

    @staticmethod
    def saveMetadata(instance, path, sc):
        """
        Saves metadata + Params to: path + "/metadata"

        - class
        - timestamp
        - sparkVersion
        - uid
        - paramMap
        - defaultParamMap (since 2.4.0)
        - (optionally, extra metadata)

        Parameters
        ----------
        extraMetadata : dict, optional
            Extra metadata to be saved at same level as uid, paramMap, etc.
        paramMap : dict, optional
            If given, this is saved in the "paramMap" field.
        """
        metadataPath = os.path.join(path, "metadata")
        metadataJson = СommonPickleMLWriter._get_metadata_to_save(instance,
                                                                  sc)
        sc.parallelize([metadataJson], 1).saveAsTextFile(metadataPath)

    @staticmethod
    def _get_metadata_to_save(instance, sc):
        """
        Helper for :py:meth:`СommonPickleMLWriter.saveMetadata` which extracts the JSON to save.
        This is useful for ensemble models which need to save metadata for many sub-models.

        Notes
        -----
        See :py:meth:`DefaultParamsWriter.saveMetadata` for details on what this includes.
        """
        uid = instance.uid
        cls = instance.__module__ + '.' + instance.__class__.__name__

        basicMetadata = {"class": cls, "timestamp": int(round(time.time() * 1000)),
                         "sparkVersion": sc.version, "uid": uid, "paramMap": None,
                         "defaultParamMap": None}

        return json.dumps(basicMetadata, separators=[',', ':'])


class СommonPickleMLReader(MLReader):

    def load(self, path):
        """Load the ML instance from the input path."""

        with open(os.path.join(path, "transformer_class_instance.pickle"), 'rb') as handle:
            instance = pickle.load(handle)

        return instance


class SparkLabelEncoderTransformerMLWritable(MLWritable):
    def write(self) -> MLWriter:
        "Returns MLWriter instance that can save the Transformer instance."
        return SparkLabelEncoderTransformerMLWriter(self)


class SparkLabelEncoderTransformerMLReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return SparkLabelEncoderTransformerMLReader()


class SparkLabelEncoderTransformerMLWriter(MLWriter):
    """Implements saving an Estimator/Transformer instance to disk.
    Used when saving a trained pipeline.
    Implements MLWriter.saveImpl(path) method.
    """

    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        logger.info(f"Save {self.instance.__class__.__name__} to '{path}'")

        СommonPickleMLWriter.saveMetadata(self.instance, path, self.sc)

        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "transformer_class_instance.pickle"), 'wb') as handle:
            indexer_model = self.instance.indexer_model
            self.instance.indexer_model = None
            pickle.dump(self.instance, handle)
            self.instance.indexer_model = indexer_model

        self.instance.indexer_model.write().overwrite().save(os.path.join(path, "indexer_model"))


class SparkLabelEncoderTransformerMLReader(MLReader):

    def load(self, path):
        """Load the ML instance from the input path."""

        with open(os.path.join(path, "transformer_class_instance.pickle"), 'rb') as handle:
            instance = pickle.load(handle)

        from lightautoml.spark.transformers.scala_wrappers.laml_string_indexer import LAMLStringIndexerModel
        indexer_model = LAMLStringIndexerModel.load(os.path.join(path, "indexer_model"))
        instance.indexer_model = indexer_model

        return instance


class LightGBMModelWrapperMLWriter(MLWriter):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        logger.info(f"Save {self.instance.__class__.__name__} to '{path}'")

        LightGBMModelWrapperMLWriter.saveMetadata(self.instance, path, self.sc)

        model: Union[LightGBMRegressionModel, LightGBMClassificationModel] = self.instance.model
        model.saveNativeModel(os.path.join(path, "model"))

    @staticmethod
    def saveMetadata(instance, path, sc):
        """
        Saves metadata + Params to: path + "/metadata"

        - class
        - timestamp
        - sparkVersion
        - uid
        - paramMap
        - defaultParamMap (since 2.4.0)
        - (optionally, extra metadata)

        Parameters
        ----------
        extraMetadata : dict, optional
            Extra metadata to be saved at same level as uid, paramMap, etc.
        paramMap : dict, optional
            If given, this is saved in the "paramMap" field.
        """
        metadataPath = os.path.join(path, "metadata")
        metadataJson = LightGBMModelWrapperMLWriter._get_metadata_to_save(instance,
                                                                          sc)
        sc.parallelize([metadataJson], 1).saveAsTextFile(metadataPath)

    @staticmethod
    def _get_metadata_to_save(instance, sc):
        """
        Helper for :py:meth:`LightGBMModelWrapperMLWriter.saveMetadata` which extracts the JSON to save.
        This is useful for ensemble models which need to save metadata for many sub-models.

        Notes
        -----
        See :py:meth:`LightGBMModelWrapperMLWriter.saveMetadata` for details on what this includes.
        """
        uid = instance.uid
        cls = instance.__module__ + '.' + instance.__class__.__name__
        model_cls = instance.model.__module__ + '.' + instance.model.__class__.__name__

        if isinstance(instance.model, LightGBMClassificationModel):
            rawPredictionCol = instance.model.getRawPredictionCol()
        else:
            rawPredictionCol = None
        basicMetadata = {"class": cls, "timestamp": int(round(time.time() * 1000)),
                         "sparkVersion": sc.version, "uid": uid, "paramMap":
                         {"featuresCol": instance.model.getFeaturesCol(),
                          "predictionCol": instance.model.getPredictionCol(),
                          "rawPredictionCol": rawPredictionCol
                         },
                         "defaultParamMap": None, "modelClass": model_cls}

        return json.dumps(basicMetadata, separators=[',', ':'])


class LightGBMModelWrapperMLReader(MLReader):

    def load(self, path):
        """Load the ML instance from the input path and wrap by LightGBMModelWrapper()"""

        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if metadata["modelClass"].endswith('LightGBMRegressionModel'):
            from synapse.ml.lightgbm.LightGBMRegressionModel import (
                LightGBMRegressionModel as model_type,
            )
        elif metadata["modelClass"].endswith('LightGBMClassificationModel'):
            from synapse.ml.lightgbm.LightGBMClassificationModel import (
                LightGBMClassificationModel as model_type,
            )
        else:
            raise NotImplementedError("Unknown model type.")

        from lightautoml.spark.ml_algo.boost_lgbm import LightGBMModelWrapper
        model_wrapper = LightGBMModelWrapper()
        model_wrapper.model = model_type.loadNativeModelFromFile(os.path.join(path, "model"))
        model_wrapper.model.setFeaturesCol(metadata["paramMap"]["featuresCol"])
        model_wrapper.model.setPredictionCol(metadata["paramMap"]["predictionCol"])
        if metadata["paramMap"]["rawPredictionCol"]:
            model_wrapper.model.setRawPredictionCol(metadata["paramMap"]["rawPredictionCol"])

        return model_wrapper


@inherit_doc
class LAMLStringIndexerModelJavaMLReadable(MLReadable):
    """
    (Private) Mixin for instances that provide JavaMLReader.
    """

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return LAMLStringIndexerModelJavaMLReader(cls)


def _jvm():
    """
    Returns the JVM view associated with SparkContext. Must be called
    after SparkContext is initialized.
    """
    jvm = SparkContext._jvm
    if jvm:
        return jvm
    else:
        raise AttributeError("Cannot load _jvm from SparkContext. Is SparkContext initialized?")


@inherit_doc
class LAMLStringIndexerModelJavaMLReader(MLReader):
    """
    (Private) Specialization of :py:class:`MLReader` for :py:class:`JavaParams` types
    """

    def __init__(self, clazz):
        super(LAMLStringIndexerModelJavaMLReader, self).__init__()
        self._clazz = clazz
        self._jread = self._load_java_obj(clazz).read()

    def load(self, path):
        """Load the ML instance from the input path."""
        if not isinstance(path, str):
            raise TypeError("path should be a string, got type %s" % type(path))
        java_obj = self._jread.load(path)
        if not hasattr(self._clazz, "_from_java"):
            raise NotImplementedError("This Java ML type cannot be loaded into Python currently: %r"
                                      % self._clazz)
        return self._clazz._from_java(java_obj)

    def session(self, sparkSession):
        """Sets the Spark Session to use for loading."""
        self._jread.session(sparkSession._jsparkSession)
        return self

    @classmethod
    def _load_java_obj(cls, clazz):
        """Load the peer Java object of the ML instance."""
        java_class = "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexerModel"
        java_obj = _jvm()
        for name in java_class.split("."):
            java_obj = getattr(java_obj, name)
        return java_obj