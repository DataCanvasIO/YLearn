"""
Beside effect score which measures the ability of estimating the causal
effect, we should also implement training_score which can measure
performances of machine learning models.
"""
import torch
from .base_models import BaseEstLearner