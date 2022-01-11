import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

np.random.seed(2022)


class BaseEstimationMethod:

    def __init__(self, ml_model='LR', new_ml_model=None,):
        self.ml_model_dic = {
            'LR': LinearRegression(),
        }
        if ml_model not in self.ml_model_dic:
            self.add_model(new_ml_model, name=ml_model)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def fit_predict(self, X):
        pass

    def estimate(self, X, target='ATE'):
        pass

    def add_model(self, model, name):
        self.model_dic[name] = model
        # add some other methods
        pass


class COM(BaseEstimationMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_model = self.ml_model_dic[kwargs['ml_model']]

    def estimate(self, X, y, treatment, target='ATE'):
        self.ml_model.fit(X, y)
        if target == 'ATE':
            Xt1 = pd.DataFrame.copy(X)
            Xt1[treatment] = 1
            Xt0 = pd.DataFrame.copy(X)
            Xt0[treatment] = 0
            ate = (
                self.ml_model.predict(Xt1) -
                self.ml_model.predict(Xt0)
            ).mean()
            return ate
        elif target == 'CATE':
            pass
        elif target == 'ITE':
            pass
        elif target == 'CITE':
            pass
        else:
            print(
                'Do not support estimation of quantities other \
                    than ATE, CATE, ITE, or CITE'
            )


class GroupCOM(BaseEstimationMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_model_t1 = self.ml_model_dic[kwargs['ml_model']]
        self.ml_model_t0 = self.ml_model_dic[kwargs['ml_model']]

    def estimate(self, X, y, treatment, target='ATE'):
        t1_index = X[treatment] > 0
        t0_index = X[treatment] == 0
        X_without_treatment = X.drop([treatment], axis=1)
        self.ml_model_t1.fit(X_without_treatment.loc[t1_index], y[t1_index])
        self.ml_model_t0.fit(X_without_treatment.loc[t0_index], y[t0_index])
        if target == 'ATE':
            ate = (
                self.ml_model_t1.predict(X_without_treatment) -
                self.ml_model_t0.predict(X_without_treatment)
            ).mean()
            return ate
        elif target == 'CATE':
            pass
        elif target == 'ITE':
            pass
        elif target == 'CITE':
            pass
        else:
            print(
                'Do not support estimation of quantities other \
                    than ATE, CATE, ITE, or CITE'
            )


class PropensityScore(BaseEstimationMethod):

    def __init__(self) -> None:
        super().__init__()


class TLearner(BaseEstimationMethod):
    def __init__(self) -> None:
        super().__init__()


class XLearner(BaseEstimationMethod):
    def __init__(self) -> None:
        super().__init__()
