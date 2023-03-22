# -*- coding:utf-8 -*-
from collections import OrderedDict

import numpy as np
from sklearn import metrics as sk_metrics

from ._common import const, infer_task_type, unique


def _task_to_average(task):
    if task == const.TASK_MULTICLASS:
        average = 'macro'
    else:
        average = 'binary'
    return average


def calc_score(y_true, y_preds, y_proba=None, *,
               metrics=None, task=None, pos_label=None, classes=None, average=None):
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)

    if task is None:
        task, _ = infer_task_type(y_true)

    if metrics is None:
        if task == const.TASK_REGRESSION:
            metrics = ['mae', 'mse', 'rmse']
        else:
            metrics = ['accuracy','precision',  'recall', 'f1']

    if task == const.TASK_BINARY and pos_label is None:
        if classes is None:
            classes = list(sorted(unique(y_true)))
        assert len(classes) == 2, 'classes of binary task should have two elements.'
        pos_label = classes[-1]

    if average is None:
        average = _task_to_average(task)

    recall_options = dict(average=average, labels=classes)
    if pos_label is not None:
        recall_options['pos_label'] = pos_label

    score = OrderedDict()
    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric_lower = metric.lower()
            if metric_lower == 'auc':
                if len(y_proba.shape) == 2:
                    if task == const.TASK_MULTICLASS:
                        score[metric] = sk_metrics.roc_auc_score(y_true, y_proba, multi_class='ovo', labels=classes)
                    else:
                        score[metric] = sk_metrics.roc_auc_score(y_true, y_proba[:, 1])
                else:
                    score[metric] = sk_metrics.roc_auc_score(y_true, y_proba)
            elif metric_lower == 'accuracy':
                if y_preds is None:
                    score[metric] = 0
                else:
                    score[metric] = sk_metrics.accuracy_score(y_true, y_preds)
            elif metric_lower == 'recall':
                score[metric] = sk_metrics.recall_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'precision':
                score[metric] = sk_metrics.precision_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'f1':
                score[metric] = sk_metrics.f1_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'mse':
                score[metric] = sk_metrics.mean_squared_error(y_true, y_preds)
            elif metric_lower == 'mae':
                score[metric] = sk_metrics.mean_absolute_error(y_true, y_preds)
            elif metric_lower == 'msle':
                score[metric] = sk_metrics.mean_squared_log_error(y_true, y_preds)
            elif metric_lower in {'rmse', 'rootmeansquarederror', 'root_mean_squared_error'}:
                score[metric] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_preds))
            elif metric_lower == 'r2':
                score[metric] = sk_metrics.r2_score(y_true, y_preds)
            elif metric_lower in {'logloss', 'log_loss'}:
                score[metric] = sk_metrics.log_loss(y_true, y_proba, labels=classes)

    return score
