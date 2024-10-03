import numpy as np
from algorithms.generic import ParameterSet


class GlobalAverage(ParameterSet):
    def init(self, pd_know_ratings, meta_data, hyper_params):
        self.description = "avg"
        self.global_average_rating = pd_know_ratings["r_ui"].mean()

    def required_hyper_params(self):
        return []

    def update(self, **kwargs):
        pass

    def get(self, **kwargs):
        return self.global_average_rating

    def get_reg(self):
        return 0

    def export(self):
        return {"avg": self.global_average_rating}
