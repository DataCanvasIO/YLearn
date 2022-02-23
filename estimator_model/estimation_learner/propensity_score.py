from sklearn import linear_model

# TODO: consider treatments other than binary treatment.


class PropensityScore:
    def __init__(self, ml_model='LogisticR'):
        self.ml_model_dic = {
            'LogisticR': linear_model.LogisticRegression()
        }

        if type(ml_model) is str:
            model = self.ml_model_dic[ml_model]

        self.ml_model = model

    def fit_ps_model(self, data, treatment, adjustment):
        self.ml_model.fit(data[adjustment], data[treatment])

    def get_ps(self, data, adjustment):
        return self.ml_model.predict(data[adjustment])

    def fit_get_ps(self, train_data, treatment, adjustment, pre_data):
        self.ml_model.fit_ps_model(train_data, treatment, adjustment)
        return self.get_ps(pre_data, adjustment)


class InversePorbWeighting:
    """
    Inverse Probability Weighting. The identification equation is defined as
        E[y|do(x)] = E[I(X=x)y / P(x|W)],
    where I is the indicator function and W is the adjustment set.
    For binary treatment, we have
        ATE = E[y|do(x=1) - y|do(x=0)] = 
            E[I(X=1)y / e(W)] - E[E[I(X=0)y / (1 - e(w))]
    where e(w) is the propensity score 
        e(w) = P(x|W).
    Therefore, the final estimated ATE should be
        1 / n_1 \sum_{i| x_i = 1} y_i / e(w_i)
            - 1 / n_2 \sum_{j| x_j = 0} y_j / (1 - e(w_i)).
    """
    # TODO: support more methods.

    def __init__(self, ew_model) -> None:
        self.ew_model = ew_model

    def estimate(self, data, outcome, treatment, adjustment,
                 quantity='ATE', condition=None, condition_set=None):
        if quantity == 'ATE':
            return self.estimate_ate(data, outcome, treatment, adjustment)
        elif quantity == 'CATE':
            return self.estimate_cate(
                data, outcome, treatment, adjustment, condition, condition_set
            )
        else:
            raise NotImplementedError

    def estimate_ate(self, data, outcome, treatment, adjustment):
        self.ew_model.fit(data, treatment, adjustment)
        t1_data = data.loc[data[treatment] > 0]
        t0_data = data.loc[data[treatment] <= 0]
        t1_ew = self.ew_model.get_ps(t1_data, adjustment)
        t0_ew = self.ew_model.get_ps(t0_data, adjustment)
        result = (t1_data[outcome] / t1_ew).mean() \
            - (t0_data[outcome] / t0_ew).mean()
        return result

    def estimate_cate(self, data, outcome, treatment,
                      adjustment, condition, condition_set):
        new_data = data.loc[condition].drop(list(condition_set), axis=1)
        return self.estimate_ate(new_data, outcome, treatment, adjustment)
