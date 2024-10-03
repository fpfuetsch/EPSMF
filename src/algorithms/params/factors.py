import numpy as np
import pandas as pd

from algorithms.generic import ParameterSet
from algorithms.utils import clip_gradient, pandas_to_numpy


class StandardFactors(ParameterSet):
    def __init__(self, non_negative=False):
        self.non_negative = non_negative

    def required_hyper_params(self):
        params = ["n_factors", "reg_param_u", "reg_param_i"]
        if self.non_negative:
            return params
        else:
            return params + ["learning_rate", "init_mean", "init_std"]

    def is_non_negative(self):
        return self.non_negative

    def init(self, pd_known_ratings, meta_data, hyper_params):
        students = meta_data["students"]
        exercises = meta_data["exercises"]

        self.description = "p_u * q_i^T"
        known_ratings, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings)
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.known_ratings = known_ratings
        self.learning_rate = hyper_params.learning_rate
        self.n_factors = hyper_params.n_factors
        self.reg_param_u = hyper_params.reg_param_u
        self.reg_param_i = hyper_params.reg_param_i
        self.n_u = len(students)
        self.n_i = len(exercises)

        if hasattr(hyper_params, 'matrix_i_init'):
            self.update_only_users = True
            self.n_factors = hyper_params.matrix_i_init.shape[1] - 1
        else:
            self.update_only_users = False

        if self.non_negative:
            self.matrix_u = np.random.rand(self.n_u, self.n_factors)
            self.matrix_i = np.random.rand(self.n_i, self.n_factors)
        else:
            self.matrix_u = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (self.n_u, self.n_factors))
            self.matrix_i = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (self.n_i, self.n_factors))

        if hasattr(hyper_params, 'matrix_i_init'):
            matrix_i_init = hyper_params.matrix_i_init[hyper_params.matrix_i_init["exercise"].isin(exercises)]
            matrix_i_init.loc[:, "exercise"] = matrix_i_init["exercise"].apply(lambda x: item_to_id[x])
            matrix_i_init = matrix_i_init.sort_values("exercise", ascending=True)
            matrix_i_init = matrix_i_init.drop("exercise", axis=1).to_numpy()
            self.matrix_i = matrix_i_init

    def export(self):
        matrix_u_pd = pd.DataFrame(self.matrix_u, index=pd.Index(self.user_to_id.keys(), name="student"))
        matrix_i_pd = pd.DataFrame(self.matrix_i, index=pd.Index(self.item_to_id.keys(), name="exercise"))
        return {"matrix_u_pd": matrix_u_pd, "matrix_i_pd": matrix_i_pd}

    def get(self, user, item, t_ui):
        return np.dot(self.matrix_u[user], self.matrix_i[item])

    def get_reg(self):
        return self.reg_param_u * np.linalg.norm(self.matrix_u) ** 2 + self.reg_param_i * np.linalg.norm(self.matrix_i) ** 2

    def update(self, error, user, item, t_ui):
        self.matrix_u[user] += self.learning_rate * clip_gradient(error * self.matrix_i[item] - self.reg_param_u * self.matrix_u[user])
        if not self.update_only_users:
            self.matrix_i[item] += self.learning_rate * clip_gradient(error * self.matrix_u[user] - self.reg_param_i * self.matrix_i[item])

    def nn_init_interims(self):
        self.u_numerator = np.zeros((self.n_u, self.n_factors))
        self.u_denominator = np.zeros((self.n_u, self.n_factors))
        self.i_numerator = np.zeros((self.n_i, self.n_factors))
        self.i_denominator = np.zeros((self.n_i, self.n_factors))

    def nn_update_interims(self, user, item, t_ui, r_ui, estimation):
        self.u_numerator[user] += self.matrix_i[item] * r_ui
        self.u_denominator[user] += self.matrix_i[item] * estimation
        if not self.update_only_users:
            self.i_numerator[item] += self.matrix_u[user] * r_ui
            self.i_denominator[item] += self.matrix_u[user] * estimation

    def nn_update(self, n_rating_user, n_rating_item, n_rating_t_ui):
        for user in range(self.n_u):
            for factor in range(self.n_factors):
                # if factor is 0, it can not be changed anymore (can happen if all ratings are 0)
                if self.matrix_u[user][factor] != 0:
                    self.u_denominator[user][factor] += n_rating_user[user] * self.reg_param_u * self.matrix_u[user][factor]
                    self.matrix_u[user][factor] *= self.u_numerator[user][factor] / self.u_denominator[user][factor]
        if not self.update_only_users:
            for item in range(self.n_i):
                for factor in range(self.n_factors):
                    if self.matrix_i[item][factor] != 0:
                        self.i_denominator[item][factor] += n_rating_item[item] * self.reg_param_i * self.matrix_i[item][factor]
                        self.matrix_i[item][factor] *= self.i_numerator[item][factor] / self.i_denominator[item][factor]
