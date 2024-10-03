import numpy as np
import pandas as pd

from algorithms.generic import ParameterSet
from algorithms.utils import clip_gradient, pandas_to_numpy, calc_dev, calc_spline_function


class TimeBinBiasShift(ParameterSet):
    """Capture time bin specific variability
    - L. Xiang and Q. Yang, “Time-Dependent Models in Collaborative Filtering Based Recommender System,”
     in 2009 IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology,
     Sep. 2009, pp. 450-457. doi: 10.1109/WI-IAT.2009.78.
    """

    def __init__(self, type):
        self.type = type

    def required_hyper_params(self):
        params = ["learning_rate", "bin_size"]
        if self.type == "user":
            return params + ["reg_param_but"]
        elif self.type == "item":
            return params + ["reg_param_bit"]

    def init(self, pd_known_ratings, meta_data, hyper_params):
        students = meta_data["students"]
        exercises = meta_data["exercises"]
        days = meta_data["days"]
        n_days = days.max() + 1

        known_ratings, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings)
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.known_ratings = known_ratings

        assert hyper_params.bin_size > 0
        self.bin_size = hyper_params.bin_size
        self.learning_rate = hyper_params.learning_rate

        remaining_days = n_days % self.bin_size
        n_bins = (n_days // self.bin_size) + (1 if remaining_days > 0 else 0)
        if self.type == "user":
            self.description = "b_ut"
            self.reg_param = hyper_params.reg_param_but
            self.bt = np.zeros((len(students), n_bins))
        elif self.type == "item":
            self.description = "b_it"
            self.reg_param = hyper_params.reg_param_bit
            self.bt = np.zeros((len(exercises), n_bins))

        # knowing that before update get is called, we can cache the bin
        self.bin = None

    def update(self, error, **kwargs):
        entity = kwargs[self.type]
        self.bt[entity][self.bin] += self.learning_rate * clip_gradient(error - self.reg_param * self.bt[entity][self.bin])

    def get(self, **kwargs):
        entity = kwargs[self.type]
        self.bin = kwargs["t_ui"] // self.bin_size
        return self.bt[entity][self.bin]

    def get_reg(self):
        return self.reg_param * np.linalg.norm(self.bt) ** 2

    def export(self):
        if self.type == "user":
            return {"biases_ut_pd": pd.DataFrame(self.bt, index=pd.Index(self.user_to_id.keys(), name="student"))}
        elif self.type == "item":
            return {"biases_it_pd": pd.DataFrame(self.bt, index=pd.Index(self.item_to_id.keys(), name="exercise"))}


class TimeBinUserBiasShift(TimeBinBiasShift):
    def __init__(self):
        super().__init__("user")


class TimeBinItemBiasShift(TimeBinBiasShift):
    def __init__(self):
        super().__init__("item")


class LinearBiasShift(ParameterSet):
    """
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, type):
        self.type = type

    def required_hyper_params(self):
        return ["learning_rate", "beta", "reg_param_linear_bu", "init_mean", "init_std"]

    def init(self, pd_known_ratings, meta_data, hyper_params):
        if self.type != "user":
            raise ValueError("LinearBiasShift only supports user type")

        pd_t_u_mean = meta_data["t_u_mean"]
        students = meta_data["students"]
        exercises = meta_data["exercises"]

        known_ratings, t_u_mean, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings, pd_t_u_mean=pd_t_u_mean)
        self.known_ratings = known_ratings
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.t_mean = t_u_mean
        self.description = "alpha_bu * dev(t)"

        self.learning_rate = hyper_params.learning_rate
        self.beta = hyper_params.beta
        self.reg_param = hyper_params.reg_param_linear_bu
        self.alpha = np.random.normal(hyper_params.init_mean, hyper_params.init_std, len(students))

        # knowing that before update get is called, we can cache the t_diff
        self.t_diff = None

    def update(self, error, user, item, t_ui):
        self.alpha[user] += self.learning_rate * clip_gradient(error * self.t_diff - self.reg_param * self.alpha[user])

    def get(self, user, item, t_ui):
        self.t_diff = calc_dev(t_ui, self.t_mean[user], self.beta)
        return self.alpha[user] * self.t_diff

    def get_reg(self):
        return self.reg_param * np.sum(self.alpha**2)

    def export(self):
        return {"alpha_bu_pd": pd.Series(self.alpha, index=pd.Index(self.user_to_id.keys(), name="student"))}


class LinearUserBiasShift(LinearBiasShift):
    def __init__(self):
        super().__init__("user")


class SplineBiasShift(ParameterSet):
    """
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, type, non_negative):
        self.type = type
        self.non_negative = non_negative

    def required_hyper_params(self):
        params = ["gamma", "kernel_factor", "reg_param_spline_bu"]
        if self.non_negative:
            return params
        else:
            return params + ["learning_rate", "init_mean", "init_std"]

    def is_non_negative(self):
        return self.non_negative

    def init(self, pd_known_ratings, meta_data, hyper_params):
        if self.type != "user":
            raise ValueError("LinearBiasShift only supports user type")

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
        self.description = "spline(u, t)"
        self.reg_param = hyper_params.reg_param_spline_bu
        self.learning_rate = hyper_params.learning_rate
        self.gamma = hyper_params.gamma
        self.n_u = len(students)

        shape = np.zeros(len(students)).shape
        self.spline_factors = np.empty(shape, dtype=object)
        self.spline_kernels = np.empty(shape, dtype=object)
        for u, (t_u, t_max, n_ratings) in enumerate(zip(t_u, t_u_max, n_ratings_u)):
            n_kernel = 1 + (t_max - t_u) // hyper_params.kernel_factor
            # n_kernel = round(n_ratings**kernel_factor)
            kernels = np.arange(t_u, t_max + 1, max(1, (t_max - t_u) // n_kernel))
            self.spline_kernels[u] = kernels
            if self.non_negative:
                self.spline_factors[u] = np.random.rand(len(kernels))
            else:
                self.spline_factors[u] = np.random.normal(hyper_params.init_mean, hyper_params.init_std, len(kernels))

        # knowing that before update()/update_interims() get() is called, we can cache the derivative
        self.sp_u_deriv = None


    def get(self, user, item, t_ui):
        sp_u, self.sp_u_deriv = calc_spline_function(user, t_ui, self.spline_factors, self.spline_kernels, self.gamma)
        return sp_u

    def get_reg(self):
        return self.reg_param * np.linalg.norm(np.hstack(self.spline_factors)) ** 2

    def export(self):
        spline_factors_bu_pd = pd.Series(self.spline_factors, index=pd.Index(self.user_to_id.keys(), name="student"))
        spline_kernels_bu_pd = pd.Series(self.spline_kernels, index=pd.Index(self.user_to_id.keys(), name="student"))
        return {"spline_factors_bu_pd": spline_factors_bu_pd, "spline_kernels_bu_pd": spline_kernels_bu_pd}

    def update(self, error, user, item, t_ui):
        self.spline_factors[user] += self.learning_rate * clip_gradient(
            error * self.sp_u_deriv - self.reg_param * self.spline_factors[user]
        )

    def nn_init_interims(self):
        self.spline_numerator = np.zeros(shape=self.spline_factors.shape, dtype=object)
        self.spline_denominator = np.zeros(shape=self.spline_factors.shape, dtype=object)

    def nn_update_interims(self, user, item, t_ui, r_ui, estimation):
        self.spline_numerator[user] += self.sp_u_deriv * r_ui
        self.spline_denominator[user] += self.sp_u_deriv * estimation

    def nn_update(self, n_rating_user, n_rating_item, n_rating_t_ui):
        for user in range(self.n_u):
            for kernel in range(len(self.spline_kernels[user])):
                # if param is 0, it can not be changed anymore (can happen if all ratings are 0)
                if self.spline_factors[user][kernel] != 0:
                    self.spline_denominator[user][kernel] += n_rating_user[user] * self.reg_param * self.spline_factors[user][kernel]
                    self.spline_factors[user][kernel] *= self.spline_numerator[user][kernel] / self.spline_denominator[user][kernel]


class SplinesUserBiasShift(SplineBiasShift):
    def __init__(self, non_negative=False):
        super().__init__("user", non_negative)


class FactorBiasShift(ParameterSet):
    """
    - L. Xiang and Q. Yang, “Time-Dependent Models in Collaborative Filtering Based Recommender System,” in 2009
      IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology, Sep. 2009,
      pp. 450-457. doi: 10.1109/WI-IAT.2009.78.
    """

    def __init__(self, type):
        self.type = type

    def required_hyper_params(self):
        params = ["learning_rate", "init_mean", "init_std", "n_factors"]
        if self.type == "user":
            return params + ["reg_param_user_bias_shift_x", "reg_param_user_bias_shift_z"]
        elif self.type == "item":
            return params + ["reg_param_user_bias_shift_s", "reg_param_user_bias_shift_y"]

    def init(self, pd_known_ratings, meta_data, hyper_params):
        students = meta_data["students"]
        exercises = meta_data["exercises"]
        days = meta_data["days"]
        self.taus = meta_data["taus"]
        self.omegas = meta_data["omegas"]
        pd_t_u = meta_data["t_u"]
        pd_t_i = meta_data["t_i"]

        known_ratings, t_u, t_i, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings, pd_t_u, pd_t_i)
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id

        self.learning_rate = hyper_params.learning_rate

        if self.type == "user":
            self.description = "x_u^T * z_tau"
            self.t_begin = t_u
            self.reg_param_entity = hyper_params.reg_param_user_bias_shift_x
            self.reg_param_temporal = hyper_params.reg_param_user_bias_shift_z
            self.matrix_entity = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (len(students), hyper_params.n_factors))
            self.matrix_temporal = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (days.max() + 1, hyper_params.n_factors))
        elif self.type == "item":
            self.description = "s_i^T * y_omega"
            self.t_begin = t_i
            self.reg_param_entity = hyper_params.reg_param_user_bias_shift_s
            self.reg_param_temporal = hyper_params.reg_param_user_bias_shift_y
            self.matrix_entity = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (len(exercises), hyper_params.n_factors))
            self.matrix_temporal = np.random.normal(hyper_params.init_mean, hyper_params.init_std, (days.max() + 1, hyper_params.n_factors))

    def update(self, error, **kwargs):
        entity = kwargs[self.type]
        delta_t = kwargs["t_ui"] - self.t_begin[entity]
        self.matrix_entity[entity] += self.learning_rate * clip_gradient(
            error * self.matrix_temporal[delta_t] - self.reg_param_entity * self.matrix_entity[entity]
        )
        self.matrix_temporal[delta_t] += self.learning_rate * clip_gradient(
            error * self.matrix_entity[entity] - self.reg_param_temporal * self.matrix_temporal[delta_t]
        )

    def get(self, **kwargs):
        entity = kwargs[self.type]
        delta_t = kwargs["t_ui"] - self.t_begin[entity]
        return np.dot(self.matrix_entity[entity], self.matrix_temporal[delta_t])

    def get_reg(self):
        return (
            self.reg_param_entity * np.linalg.norm(self.matrix_entity) ** 2
            + self.reg_param_temporal * np.linalg.norm(self.matrix_temporal) ** 2
        )

    def export(self):
        if self.type == "user":
            matrix_x_pd = pd.DataFrame(self.matrix_entity, index=pd.Index(self.user_to_id.keys(), name="student"))
            matrix_z_pd = pd.DataFrame(self.matrix_temporal)
            matrix_z_pd = matrix_z_pd[matrix_z_pd.index.isin(self.taus)]
            return {"matrix_x_pd": matrix_x_pd, "matrix_z_pd": matrix_z_pd}
        elif self.type == "item":
            matrix_s_pd = pd.DataFrame(self.matrix_entity, index=pd.Index(self.item_to_id.keys(), name="exercise"))
            matrix_y_pd = pd.DataFrame(self.matrix_temporal)
            matrix_y_pd = matrix_y_pd[matrix_y_pd.index.isin(self.omegas)]
            return {"matrix_s_pd": matrix_s_pd, "matrix_y_pd": matrix_y_pd}


class FactorUserBiasShift(FactorBiasShift):
    def __init__(self):
        super().__init__("user")


class FactorItemBiasShift(FactorBiasShift):
    def __init__(self):
        super().__init__("item")
