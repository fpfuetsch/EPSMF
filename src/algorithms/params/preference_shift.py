import numpy as np
import pandas as pd
from algorithms.generic import ParameterSet
from algorithms.utils import clip_gradient, pandas_to_numpy, calc_dev, calc_spline_function


class LinearUserPreferenceShiftFactors(ParameterSet):
    """
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def init(self, pd_known_ratings, meta_data, hyper_params):
        pd_t_u_mean = meta_data["t_u_mean"]
        students = meta_data["students"]
        exercises = meta_data["exercises"]

        known_ratings, t_u_mean, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings, pd_t_u_mean=pd_t_u_mean)
        self.known_ratings = known_ratings
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.t_u_mean = t_u_mean
        self.description = "(p_u + alpha_u * dev(t)) * q_i^T"
        self.learning_rate = hyper_params.learning_rate
        self.beta = hyper_params.beta
        self.reg_param_alpha = hyper_params.reg_param_linear_u
        self.reg_param_u = hyper_params.reg_param_u
        self.reg_param_i = hyper_params.reg_param_i

        self.alpha = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (len(students), hyper_params.n_factors))
        self.matrix_u = np.random.normal(
            hyper_params.init_mean, hyper_params.init_std, (len(meta_data["students"]), hyper_params.n_factors)
        )
        self.matrix_i = np.random.normal(
            hyper_params.init_mean, hyper_params.init_std, (len(meta_data["exercises"]), hyper_params.n_factors)
        )

        # knowing that before update get is called, we can cache the t_diff
        self.t_diff = None

    def required_hyper_params(self):
        return ["learning_rate", "n_factors", "reg_param_u", "reg_param_i", "reg_param_linear_u", "beta", "init_mean", "init_std"]

    def update(self, error, user, item, t_ui):
        # matrix factorization part
        self.matrix_u[user] += self.learning_rate * clip_gradient(error * self.matrix_i[item] - self.reg_param_u * self.matrix_u[user])
        self.matrix_i[item] += self.learning_rate * clip_gradient(
            error * (self.matrix_u[user] + self.alpha[user] * self.t_diff) - self.reg_param_i * self.matrix_i[item]
        )
        # preference shift part
        self.alpha[user] += self.learning_rate * clip_gradient(
            error * self.t_diff * self.matrix_i[item] - self.reg_param_alpha * self.alpha[user]
        )

    def get(self, user, item, t_ui):
        self.t_diff = calc_dev(t_ui, self.t_u_mean[user], self.beta)
        return np.dot(self.matrix_u[user] + self.alpha[user] * self.t_diff, self.matrix_i[item])

    def get_reg(self):
        return (
            self.reg_param_u * np.linalg.norm(self.matrix_u) ** 2
            + self.reg_param_alpha * np.linalg.norm(self.alpha) ** 2
            + self.reg_param_i * np.linalg.norm(self.matrix_i) ** 2
        )

    def export(self):
        matrix_u_pd = pd.DataFrame(self.matrix_u, index=pd.Index(self.user_to_id.keys(), name="student"))
        alpha_u_pd = pd.DataFrame(self.alpha, index=pd.Index(self.user_to_id.keys(), name="student"))
        matrix_i_pd = pd.DataFrame(self.matrix_i, index=pd.Index(self.item_to_id.keys(), name="exercise"))
        return {"matrix_u_pd": matrix_u_pd, "alpha_u_pd": alpha_u_pd, "matrix_i_pd": matrix_i_pd}


class SplineUserPreferenceShiftFactors(ParameterSet):
    """
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, non_negative=False):
        self.non_negative = non_negative

    def is_non_negative(self):
        return self.non_negative

    def required_hyper_params(self):
        params = ["n_factors", "reg_param_u", "reg_param_i", "reg_param_spline_u", "gamma", "kernel_factor"]
        if self.non_negative:
            return params
        else:
            return params + ["learning_rate", "init_mean", "init_std"]

    def init(self, pd_known_ratings, meta_data, hyper_params):
        pd_t_u = meta_data["t_u"]
        pd_t_u_max = meta_data["t_u_max"]
        pd_n_ratings_u = meta_data["n_ratings_u"]
        students = meta_data["students"]
        exercises = meta_data["exercises"]

        known_ratings, t_u, t_u_max, n_ratings_u, user_to_id, item_to_id = pandas_to_numpy(
            students, exercises, pd_known_ratings, pd_t_u=pd_t_u, pd_t_u_max=pd_t_u_max, pd_n_ratings_u=pd_n_ratings_u
        )
        self.known_ratings = known_ratings
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.description = "(p_u + spline_u(u, t))^T * q_i"
        self.reg_param_splines = hyper_params.reg_param_spline_u
        self.reg_param_u = hyper_params.reg_param_u
        self.reg_param_i = hyper_params.reg_param_i
        self.learning_rate = hyper_params.learning_rate
        self.n_factors = hyper_params.n_factors
        self.gamma = hyper_params.gamma
        self.n_u = len(students)
        self.n_i = len(exercises)

        if hasattr(hyper_params, 'n_kernel'):
            self.n_kernel = hyper_params.n_kernel
        else:
            self.n_kernel = None
        if hasattr(hyper_params, 'matrix_i_init'):
            self.update_only_users = True
            self.n_factors = hyper_params.matrix_i_init.shape[1] - 1
        else:
            self.update_only_users = False

        if hasattr(hyper_params, 'omit_static_u'):
            self.has_static_u = not hyper_params.omit_static_u
        else:
            self.has_static_u = True

        self.spline_factors = np.empty(self.n_u, dtype=object)
        self.spline_kernels = np.empty(self.n_u, dtype=object)
        for u, (t_u, t_max, n_ratings) in enumerate(zip(t_u, t_u_max, n_ratings_u)):
            if self.n_kernel is not None:
                n_kernel = self.n_kernel
            else:
                n_kernel = 1 + (t_max - t_u) // hyper_params.kernel_factor
                # n_kernel = round(n_ratings**kernel_factor)
            kernels = np.arange(t_u, t_max + 1, max(1, (t_max - t_u) // n_kernel))
            self.spline_kernels[u] = kernels
            if self.non_negative:
                self.spline_factors[u] =  np.array([np.random.rand(len(kernels)) for _ in range(hyper_params.n_factors)])
            else:
                self.spline_factors[u] = np.array([np.random.normal(hyper_params.init_mean, hyper_params.init_std, len(kernels)) for _ in range(hyper_params.n_factors)])

        if self.non_negative:
            self.matrix_u = np.random.rand(self.n_u, self.n_factors)
            self.matrix_i = np.random.rand(self.n_i, self.n_factors)
        else:
            self.matrix_u = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (self.n_u, self.n_factors))
            self.matrix_i = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (self.n_i, self.n_factors))

        if self.update_only_users:
            matrix_i_init = hyper_params.matrix_i_init[hyper_params.matrix_i_init["exercise"].isin(exercises)]
            matrix_i_init.loc[:, "exercise"] = matrix_i_init["exercise"].apply(lambda x: item_to_id[x])
            matrix_i_init = matrix_i_init.sort_values("exercise", ascending=True)
            matrix_i_init = matrix_i_init.drop("exercise", axis=1).to_numpy()
            self.matrix_i = matrix_i_init

        if not self.has_static_u:
            self.description = "spline_u(u, t) * q_i^T"
            self.matrix_u = np.zeros((self.n_u, self.n_factors))
        else:
            self.description = "(p_u + spline_u(u, t))^T * q_i"

        # knowing that before update get is called, we can cache the sp_u and sp_u_deriv
        self.sp_u_deriv = None
        self.sp_u = None

    def export(self):
        matrix_u_pd = pd.DataFrame(self.matrix_u, index=pd.Index(self.user_to_id.keys(), name="student"))
        matrix_i_pd = pd.DataFrame(self.matrix_i, index=pd.Index(self.item_to_id.keys(), name="exercise"))
        spline_factors_u_pd = pd.Series(self.spline_factors, index=pd.Index(self.user_to_id.keys(), name="student"))
        spline_kernels_u_pd = pd.Series(self.spline_kernels, index=pd.Index(self.user_to_id.keys(), name="student"))
        return {
            "matrix_u_pd": matrix_u_pd,
            "matrix_i_pd": matrix_i_pd,
            "spline_factors_u_pd": spline_factors_u_pd,
            "spline_kernels_u_pd": spline_kernels_u_pd,
        }

    def get(self, user, item, t_ui):
        self.sp_u, self.sp_u_deriv = calc_spline_function(user, t_ui, self.spline_factors, self.spline_kernels, self.gamma, is_for_factors=True)
        return np.dot(self.matrix_u[user] + self.sp_u, self.matrix_i[item])

    def get_reg(self):
        return (
            self.reg_param_u * np.linalg.norm(self.matrix_u) ** 2
            + self.reg_param_splines * np.linalg.norm(np.hstack(self.spline_factors)) ** 2
            + self.reg_param_i * np.linalg.norm(self.matrix_i) ** 2
        )

    def update(self, error, user, item, t_ui):
        # static factorization part
        if self.has_static_u:
            self.matrix_u[user] += self.learning_rate * clip_gradient(error * self.matrix_i[item] - self.reg_param_u * self.matrix_u[user])
        if not self.update_only_users:
            self.matrix_i[item] += self.learning_rate * clip_gradient(
                error * (self.matrix_u[user] + self.sp_u) - self.reg_param_i * self.matrix_i[item]
            )
        # dynamic user factor part
        self.sp_u_deriv = np.repeat([self.sp_u_deriv], self.n_factors, axis=0) * self.matrix_i[item, np.newaxis].T
        self.spline_factors[user] += self.learning_rate * clip_gradient(
            error * self.sp_u_deriv - self.reg_param_splines * self.spline_factors[user]
        )

    def nn_init_interims(self):
        self.u_numerator = np.zeros((self.n_u, self.n_factors))
        self.u_denominator = np.zeros((self.n_u, self.n_factors))
        self.i_numerator = np.zeros((self.n_i, self.n_factors))
        self.i_denominator = np.zeros((self.n_i, self.n_factors))
        self.spline_numerator = np.zeros(shape=self.spline_factors.shape, dtype=object)
        self.spline_denominator = np.zeros(shape=self.spline_factors.shape, dtype=object)

    def nn_update_interims(self, user, item, t_ui, r_ui, estimation):
        if self.has_static_u:
            self.u_numerator[user] += self.matrix_i[item] * r_ui
            self.u_denominator[user] += self.matrix_i[item] * estimation
        if not self.update_only_users:
            self.i_numerator[item] += (self.matrix_u[user] + self.sp_u) * r_ui
            self.i_denominator[item] += (self.matrix_u[user] + self.sp_u) * estimation
        self.sp_u_deriv = np.repeat([self.sp_u_deriv], self.n_factors, axis=0) * self.matrix_i[item, np.newaxis].T
        self.spline_numerator[user] += self.sp_u_deriv * r_ui
        self.spline_denominator[user] += self.sp_u_deriv * estimation

    def nn_update(self, n_rating_user, n_rating_item, n_rating_t_ui):
        if self.has_static_u:
            for user in range(self.n_u):
                for factor in range(self.n_factors):
                        # if factor is 0, it can not be changed anymore (can happen if all ratings are 0)
                        if self.matrix_u[user][factor] != 0:
                            self.u_denominator[user][factor] += n_rating_user[user] * self.reg_param_u * self.matrix_u[user][factor]
                            self.matrix_u[user][factor] *= self.u_numerator[user][factor] / self.u_denominator[user][factor]
        for user in range(self.n_u):
            for factor in range(self.n_factors):
                for kernel in range(len(self.spline_kernels[user])):
                    # if param is 0, it can not be changed anymore (can happen if all ratings are 0)
                    if self.spline_factors[user][factor][kernel] != 0:
                        self.spline_denominator[user][factor][kernel] += n_rating_user[user] * self.reg_param_splines * self.spline_factors[user][factor][kernel]
                        self.spline_factors[user][factor][kernel] *= self.spline_numerator[user][factor][kernel] / self.spline_denominator[user][factor][kernel]
        if not self.update_only_users:
            for item in range(self.n_i):
                for factor in range(self.n_factors):
                    if self.matrix_i[item][factor] != 0:
                        self.i_denominator[item][factor] += n_rating_item[item] * self.reg_param_i * self.matrix_i[item][factor]
                        self.matrix_i[item][factor] *= self.i_numerator[item][factor] / self.i_denominator[item][factor]


class FactorPreferenceShift(ParameterSet):
    """
    - L. Xiang and Q. Yang, “Time-Dependent Models in Collaborative Filtering Based Recommender System,” in 2009
      IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology, Sep. 2009,
      pp. 450-457. doi: 10.1109/WI-IAT.2009.78.
    """

    def init(self, pd_known_ratings, meta_data, hyper_params):
        self.description = "sum(g_u * l_i * h_tau)"
        students = meta_data["students"]
        exercises = meta_data["exercises"]
        days = meta_data["days"]
        taus = meta_data["taus"]
        pd_t_u = meta_data["t_u"]
        pd_t_i = meta_data["t_i"]

        # pandas to numpy
        known_ratings, t_u, t_i, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings, pd_t_u, pd_t_i)
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.known_ratings = known_ratings
        self.t_u = t_u
        self.taus = taus

        self.learning_rate = hyper_params.learning_rate
        self.reg_param_g = hyper_params.reg_param_prefshift_g
        self.reg_param_l = hyper_params.reg_param_prefshift_l
        self.reg_param_h = hyper_params.reg_param_prefshift_h
        self.matrix_g = np.random.normal(
            hyper_params.init_mean, hyper_params.init_std, (len(meta_data["students"]), hyper_params.n_factors)
        )
        self.matrix_l = np.random.normal(
            hyper_params.init_mean, hyper_params.init_std, (len(meta_data["exercises"]), hyper_params.n_factors)
        )
        self.matrix_h = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (days.max() + 1, hyper_params.n_factors))

    def required_hyper_params(self):
        return [
            "learning_rate",
            "n_factors",
            "reg_param_prefshift_g",
            "reg_param_prefshift_l",
            "reg_param_prefshift_h",
            "init_mean",
            "init_std",
        ]

    def update(self, error, user, item, t_ui):
        tau = t_ui - self.t_u[user]
        self.matrix_g[user] += self.learning_rate * clip_gradient(
            error * self.matrix_l[item] * self.matrix_h[tau] - self.reg_param_g * self.matrix_g[user]
        )
        self.matrix_l[item] += self.learning_rate * clip_gradient(
            error * self.matrix_g[user] * self.matrix_h[tau] - self.reg_param_l * self.matrix_l[item]
        )
        self.matrix_h[tau] += self.learning_rate * clip_gradient(
            error * self.matrix_g[user] * self.matrix_l[item] - self.reg_param_h * self.matrix_h[tau]
        )

    def get(self, user, item, t_ui):
        tau = t_ui - self.t_u[user]
        return np.sum(self.matrix_g[user] * self.matrix_l[item] * self.matrix_h[tau])

    def get_reg(self):
        return (
            self.reg_param_g * np.linalg.norm(self.matrix_g) ** 2
            + self.reg_param_l * np.linalg.norm(self.matrix_l) ** 2
            + self.reg_param_h * np.linalg.norm(self.matrix_h) ** 2
        )

    def export(self):
        matrix_g_pd = pd.DataFrame(self.matrix_g, index=pd.Index(self.user_to_id.keys(), name="student"))
        matrix_l_pd = pd.DataFrame(self.matrix_l, index=pd.Index(self.item_to_id.keys(), name="exercise"))
        matrix_h_pd = pd.DataFrame(self.matrix_h)
        matrix_h_pd = matrix_h_pd[matrix_h_pd.index.isin(self.taus)]
        return {"matrix_g_pd": matrix_g_pd, "matrix_l_pd": matrix_l_pd, "matrix_h_pd": matrix_h_pd}
