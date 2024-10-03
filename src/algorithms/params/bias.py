import numpy as np
import pandas as pd
from algorithms.generic import ParameterSet
from algorithms.utils import clip_gradient, pandas_to_numpy


class Bias(ParameterSet):
    def __init__(self, type, non_negative):
        self.type = type
        self.non_negative = non_negative

    def required_hyper_params(self):
        params = ["learning_rate"]
        if self.type == "user":
            return params + ["reg_param_bu"]
        elif self.type == "item":
            return params + ["reg_param_bi"]
        elif self.type == "t_ui":
            return params + ["reg_param_bt"]

    def is_non_negative(self):
        return self.non_negative

    def init(self, pd_known_ratings, meta_data, hyper_params):
        students = meta_data["students"]
        exercises = meta_data["exercises"]
        days = meta_data["days"]
        n_days = days.max() + 1
        n_u = len(students)
        n_i = len(exercises)

        known_ratings, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings)
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.known_ratings = known_ratings
        self.learning_rate = hyper_params.learning_rate

        param_initializer = np.random.rand if self.non_negative else np.zeros
        if self.type == "user":
            self.description = "b_u"
            self.reg_param = hyper_params.reg_param_bu
            self.b = param_initializer(n_u)
            self.n = n_u
        elif self.type == "item":
            self.description = "b_i"
            self.reg_param = hyper_params.reg_param_bi
            self.b = param_initializer(n_i)
            self.n = n_i
        elif self.type == "t_ui":
            self.description = "b_t"
            self.reg_param = hyper_params.reg_param_bt
            self.b = param_initializer(n_days)
            self.n = n_days

    def export(self):
        if self.type == "user":
            return {"biases_u_pd": pd.Series(self.b, index=pd.Index(self.user_to_id.keys(), name="student"))}
        elif self.type == "item":
            return {"biases_i_pd": pd.Series(self.b, index=pd.Index(self.item_to_id.keys(), name="exercise"))}
        elif self.type == "t_ui":
            return {"biases_t_pd": pd.Series(self.b, name="t_ui")}

    def get(self, **kwargs):
        entity = kwargs[self.type]
        return self.b[entity]

    def get_reg(self):
        return self.reg_param * np.sum(self.b**2)

    def update(self, error, **kwargs):
        entity = kwargs[self.type]
        self.b[entity] += self.learning_rate * clip_gradient(error - self.reg_param * self.b[entity])

    def nn_init_interims(self):
        self.numerator = np.zeros(self.n)
        self.denominator = np.zeros(self.n)

    def nn_update_interims(self, r_ui, estimation, **kwargs):
        entity = kwargs[self.type]
        self.numerator[entity] += r_ui
        self.denominator[entity] += estimation

    def nn_update(self, **kwargs):
        n_ratings = kwargs[f"n_rating_{self.type}"]
        for entity in range(self.n):
            # if bias is 0, it can not be changed anymore (can happen if all ratings are 0), n_ratings can be 0 for time bias
            if self.b[entity] != 0 and n_ratings[entity] != 0:
                self.denominator[entity] += n_ratings[entity] * self.reg_param * self.b[entity]
                self.b[entity] *= self.numerator[entity] / self.denominator[entity]


class UserBias(Bias):
    def __init__(self, non_negative=False):
        super().__init__("user", non_negative)


class ItemBias(Bias):
    def __init__(self, non_negative=False):
        super().__init__("item", non_negative)


class TimeBias(Bias):
    def __init__(self, non_negative=False):
        super().__init__("t_ui", non_negative)
