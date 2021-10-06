#!/usr/bin/env python
# coding: utf-8

"""
2 level stacking using AutoML class with different algos on first level including LGBM, Linear and LinearL1
"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.automl.blend import MeanBlender
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearL1CD
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.pipelines.selection.linear_selector import HighCorrRemoval
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


np.random.seed(42)

print("Load data...")
data = pd.read_csv("./data/sampled_app_train.csv")
print("Data loaded")

print("Features modification from user side...")
data["BIRTH_DATE"] = (
    np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
).astype(str)
data["EMP_DATE"] = (
    np.datetime64("2018-01-01")
    + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
).astype(str)

data["report_dt"] = np.datetime64("2018-01-01")

data["constant"] = 1
data["allnan"] = np.nan

data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

data["TARGET"] = data["TARGET"]
print("Features modification finished")

print("Split data...")
train, test = train_test_split(data, test_size=0.2, random_state=42)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print(
    "Data splitted. Parts sizes: train_data = {}, test_data = {}".format(
        train.shape, test.shape
    )
)

print("Start creation selector_0...")
feat_sel_0 = LGBSimpleFeatures()
mod_sel_0 = BoostLGBM()
imp_sel_0 = ModelBasedImportanceEstimator()
selector_0 = ImportanceCutoffSelector(feat_sel_0, mod_sel_0, imp_sel_0, cutoff=0)
print("End creation selector_0...")

print("Start creation gbm_0...")
feats_gbm_0 = LGBAdvancedPipeline()
gbm_0 = BoostLGBM()
gbm_1 = BoostLGBM()
tuner_0 = OptunaTuner(n_trials=100, timeout=30, fit_on_holdout=True)
gbm_lvl0 = MLPipeline(
    [(gbm_0, tuner_0), gbm_1],
    pre_selection=selector_0,
    features_pipeline=feats_gbm_0,
    post_selection=None,
)
print("End creation gbm_0...")

print("Start creation reg_0...")
feats_reg_0 = LinearFeatures(output_categories=True)
reg_0 = LinearLBFGS()
reg_lvl0 = MLPipeline(
    [reg_0],
    pre_selection=None,
    features_pipeline=feats_reg_0,
    post_selection=HighCorrRemoval(corr_co=1),
)
print("End creation reg_0...")

print("Start creation composed selector...")
feat_sel_1 = LGBSimpleFeatures()
mod_sel_1 = BoostLGBM()
imp_sel_1 = NpPermutationImportanceEstimator()
selector_1 = NpIterativeFeatureSelector(
    feat_sel_1, mod_sel_1, imp_sel_1, feature_group_size=1
)
print("End creation composed selector...")

print("Start creation reg_l1_0...")
feats_reg_1 = LinearFeatures(output_categories=False)
reg_1 = LinearL1CD()
reg_l1_lvl0 = MLPipeline(
    [reg_1],
    pre_selection=selector_1,
    features_pipeline=feats_reg_1,
    post_selection=HighCorrRemoval(),
)
print("End creation reg_l1_0...")

print("Start creation blending...")
feats_reg_2 = LinearFeatures(output_categories=True)
reg_2 = LinearLBFGS()
reg_lvl1 = MLPipeline(
    [reg_2],
    pre_selection=None,
    features_pipeline=feats_reg_2,
    post_selection=HighCorrRemoval(corr_co=1),
)
print("End creation blending...")

print("Start creation automl...")
reader = PandasToPandasReader(
    Task(
        "binary",
    ),
    samples=None,
    max_nan_rate=1,
    max_constant_rate=1,
)

automl = AutoML(
    reader,
    [
        [gbm_lvl0, reg_lvl0, reg_l1_lvl0],
        [reg_lvl1],
    ],
    skip_conn=False,
    blender=MeanBlender(),
)
print("End creation automl...")

print("Start fit automl...")
roles = {
    "target": "TARGET",
    DatetimeRole(base_date=True, seasonality=(), base_feats=False): "report_dt",
}

oof_pred = automl.fit_predict(train, roles=roles)
print("End fit automl...")

test_pred = automl.predict(test)
print("Prediction for test data:\n{}\nShape = {}".format(test_pred, test_pred.shape))

not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

print("Check scores...")
print(
    "OOF score: {}".format(
        roc_auc_score(
            train[roles["target"]].values[not_nan], oof_pred.data[not_nan][:, 0]
        )
    )
)
print(
    "TEST score: {}".format(
        roc_auc_score(test[roles["target"]].values, test_pred.data[:, 0])
    )
)
print("Pickle automl")
with open("automl.pickle", "wb") as f:
    pickle.dump(automl, f)

print("Load pickled automl")
with open("automl.pickle", "rb") as f:
    automl = pickle.load(f)

print("Predict loaded automl")
test_pred = automl.predict(test)
print(
    "TEST score, loaded: {}".format(
        roc_auc_score(test["TARGET"].values, test_pred.data[:, 0])
    )
)

os.remove("automl.pickle")
