from sklearn import linear_model


class MlModels:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class LR(MlModels):
    def __init__(self) -> None:
        self.ml_model = linear_model.LinearRegression()

    def fit(self, X, y):
        return self.ml_model.fit(X, y)

    def predict(self, X):
        return self.ml_model.predict(X)


class NewEgModel(MlModels):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return super().fit(X, y)
