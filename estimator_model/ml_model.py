"""
Generalization of other machine learning models in the sci-kit style.
"""


class MlModels:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class LinearRegression(MlModels):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return super().fit(X, y)


class NewEgModel(MlModels):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return super().fit(X, y)
