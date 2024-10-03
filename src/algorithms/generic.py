import numpy as np
from algorithms.utils import pandas_to_numpy


class HyperParameters:
    def __init__(self, **kwargs):
        # define some default values
        self.n_iterations = 100
        self.learning_rate = 0.01
        self.init_mean = 0
        self.init_std = 1

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)


class ParameterSet:
    def init():
        pass

    def required_hyper_params():
        pass

    def is_non_negative(self):
        return False

    def get(self):
        pass

    def get_reg(self):
        pass

    def export(self):
        pass

    def update(self):
        pass

    def nn_init_interims(self):
        pass

    def nn_update_interims(self):
        pass

    def nn_update(self):
        pass


class Model:
    def __init__(self, pd_known_ratings, meta_data, hyper_params, components, verbose):
        self.pd_known_ratings = pd_known_ratings
        self.meta_data = meta_data
        self.hyper_params = hyper_params
        self.components = components
        self.verbose = verbose

        students = meta_data["students"]
        exercises = meta_data["exercises"]
        days = meta_data["days"]

        known_ratings, user_to_id, item_to_id = pandas_to_numpy(students, exercises, pd_known_ratings)
        self.known_ratings = known_ratings
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        self.n_u = len(students)
        self.n_i = len(exercises)
        self.n_t = days.max() + 1

        # check if all required hyper parameters are set
        for parameter in self.components:
            required_hyper_params = parameter.required_hyper_params()
            for param in required_hyper_params:
                if not hasattr(self.hyper_params, param):
                    raise Exception(f"Hyper parameter {param} is not set which is required by {parameter.__class__.__name__}")

        # check if all parameters are non-negative if one parameter is non-negative
        self.non_negative = any([param.is_non_negative() for param in self.components])
        if self.non_negative and not all([param.is_non_negative() for param in self.components]):
            raise Exception("All parameters must have non-negative constraint if one parameter is non-negative!")

        for parameter in self.components:
            parameter.init(self.pd_known_ratings, self.meta_data, self.hyper_params)

    def has_temporal_components(self):
        return any([any(hint in param.description for hint in ["t", "tau", "omega"]) for param in self.components])

    def regularized_square_error(self, iteration):
        error = 0
        for entry in self.known_ratings:
            user, item, t_ui, r_ui = [int(entry[0]), int(entry[1]), int(entry[2]), entry[3]]
            prediction = self._predict_internal(user, item, t_ui)
            error += (r_ui - prediction) ** 2
        for parameter in self.components:
            error += parameter.get_reg()
        print(f"Iteration: {iteration}, regularized square error: {error}")

    def fit(self):
        if self.verbose:
            description = "r_ui(t) = " + " + ".join([param.description for param in self.components])
            print(f"Fitting{' non-negative' if self.non_negative else ''} model {self.__class__.__name__}\n{description}\n")

        if self.non_negative:
            n_rating_user = np.empty(self.n_u)
            n_rating_item = np.empty(self.n_i)
            n_rating_t_ui = np.empty(self.n_t)
            for user in range(self.n_u):
                n_rating_user[user] = len(self.known_ratings[self.known_ratings[:, 0] == user])
            for item in range(self.n_i):
                n_rating_item[item] = len(self.known_ratings[self.known_ratings[:, 1] == item])
            for day in range(self.n_t):
                n_rating_t_ui[day] = len(self.known_ratings[self.known_ratings[:, 2] == day])

            for iteration in range(self.hyper_params.n_iterations):
                self.known_ratings = np.random.permutation(self.known_ratings)
                for param in self.components:
                    param.nn_init_interims()

                for entry in self.known_ratings:
                    user, item, t_ui, r_ui = [int(entry[0]), int(entry[1]), int(entry[2]), entry[3]]
                    estimation = self._predict_internal(user, item, t_ui)
                    for param in self.components:
                        param.nn_update_interims(user=user, item=item, t_ui=t_ui, r_ui=r_ui, estimation=estimation)

                for param in self.components:
                    param.nn_update(n_rating_user=n_rating_user, n_rating_item=n_rating_item, n_rating_t_ui=n_rating_t_ui)

                if self.verbose and iteration % 10 == 0:
                    self.regularized_square_error(iteration)

        else:
            for iteration in range(self.hyper_params.n_iterations):
                self.known_ratings = np.random.permutation(self.known_ratings)
                for entry in self.known_ratings:
                    user, item, t_ui, r_ui = [int(entry[0]), int(entry[1]), int(entry[2]), entry[3]]
                    estimation = self._predict_internal(user, item, t_ui)
                    error = r_ui - estimation
                    for param in self.components:
                        param.update(error=error, user=user, item=item, t_ui=t_ui)

                if self.verbose and iteration % 10 == 0:
                    self.regularized_square_error(iteration)

    def _predict_internal(self, user, item, t_ui):
        estimation = 0
        for param in self.components:
            estimation += param.get(user=user, item=item, t_ui=t_ui)
        return estimation

    def predict(self, user, item, t_ui):
        user = self.user_to_id[user]
        item = self.item_to_id[item]
        return self._predict_internal(user, item, t_ui)

    def get_params(self):
        parameters = {}
        for param in self.components:
            ex = param.export()
            for k, v in ex.items():
                parameters[k] = v
        return parameters