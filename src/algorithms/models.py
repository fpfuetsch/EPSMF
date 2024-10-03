from algorithms.generic import Model
from algorithms.params.bias_shift import FactorUserBiasShift, LinearUserBiasShift, SplinesUserBiasShift, TimeBinUserBiasShift
from algorithms.params.factors import StandardFactors
from algorithms.params.misc import GlobalAverage
from algorithms.params.bias import TimeBias, UserBias, ItemBias
from algorithms.params.preference_shift import FactorPreferenceShift, LinearUserPreferenceShiftFactors, SplineUserPreferenceShiftFactors


### BASELINE ###

class UserBiasOnly(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias()]
        super().__init__(components=components, **kwargs)


class ItemBiasOnly(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), ItemBias()]
        super().__init__(components=components, **kwargs)


class UserItemBias(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias()]
        super().__init__(components=components, **kwargs)


class UserItemTimeBias(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBias()]
        super().__init__(components=components, **kwargs)


class NN_UserItemBias(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class NN_UserItemTimeBias(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
            TimeBias(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


### Basic ###

class SVD(Model):
    """Regularized SVD algorithm.
    - S. Funk, “Netflix Challenge,” Oct. 27, 2006. https://sifter.org/simon/journal/20061027.2.html (accessed Apr. 04, 2023).
    - Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.
    """

    def __init__(self, **kwargs):
        components = [StandardFactors()]
        super().__init__(components=components, **kwargs)


class NN_SVD(Model):
    """Non Negative Regularized MF.
    - [1] X. Luo, M. Zhou, Y. Xia, and Q. Zhu, “An Efficient Non-Negative Matrix-Factorization-Based Approach to
        Collaborative Filtering for Recommender Systems,” IEEE Transactions on Industrial Informatics, vol. 10, no.
        2, pp. 1273-1284, May 2014, doi: 10.1109/TII.2014.2308433.
    """

    def __init__(self, **kwargs):
        components = [StandardFactors(non_negative=True)]
        super().__init__(components=components, **kwargs)


class SVD_UserItemBias(Model):
    """Biased regularized SVD algorithm.
    - Y. Koren, “Factorization meets the neighborhood: a multifaceted collaborative filtering components,”
      in Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, in KDD '08.
      New York, NY, USA: Association for Computing Machinery, Aug. 2008, pp. 426-434. doi: 10.1145/1401890.1401944.
    """

    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


class NN_SVD_UserItemBias(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
            StandardFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class NN_SVD_UserItemTimeBias(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
            TimeBias(non_negative=True),
            StandardFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class SVD_UserItemTimeBias(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


### Bias Shift ###

class UserItemBias_LinearUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), LinearUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemTimeBias_LinearUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBias(), LinearUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemBias_SplineUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), SplinesUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemBias_BinUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBinUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemTimeBias_SplineUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBias(), SplinesUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemBias_FactorUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), FactorUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemTimeBias_FactorUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBias(), FactorUserBiasShift()]
        super().__init__(components=components, **kwargs)


class UserItemTimeBias_BinUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), ItemBias(), TimeBias(), TimeBinUserBiasShift()]
        super().__init__(components=components, **kwargs)


class SVD_UserItemBias_LinearUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), LinearUserBiasShift(), ItemBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


class SVD_UserItemTimeBias_LinearUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), LinearUserBiasShift(), ItemBias(), TimeBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


class SVD_UserItemBias_SplineUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), SplinesUserBiasShift(), ItemBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


class SVD_UserItemTimeBias_SplineUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), SplinesUserBiasShift(), ItemBias(), TimeBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


class NN_SVD_UserItemTimeBias_SplineUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            SplinesUserBiasShift(non_negative=True),
            ItemBias(non_negative=True),
            TimeBias(non_negative=True),
            StandardFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class SVD_UserItemBias_FactorUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), FactorUserBiasShift(), ItemBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


class SVD_UserItemTimeBias_FactorUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [GlobalAverage(), UserBias(), FactorUserBiasShift(), ItemBias(), TimeBias(), StandardFactors()]
        super().__init__(components=components, **kwargs)


### Bias and Preference Shift ###

class NN_SVD_SplineUserPreferenceShift_UserItemBias(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
            SplineUserPreferenceShiftFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class NN_SVD_SplineUserPreferenceShift_ItemBias(Model):
    def __init__(self, **kwargs):
        components = [
            ItemBias(non_negative=True),
            SplineUserPreferenceShiftFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class NN_SVD_SplineUserPreferenceShift(Model):
    def __init__(self, **kwargs):
        components = [
            SplineUserPreferenceShiftFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)


class NN_SVD_SplineUserPreferenceShift_UserItemTimeBias_SplineUserBiasShift(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
            TimeBias(non_negative=True),
            SplinesUserBiasShift(non_negative=True),
            SplineUserPreferenceShiftFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)

class NN_SVD_SplineUserPreferenceShift_UserItemTimeBias(Model):
    def __init__(self, **kwargs):
        components = [
            UserBias(non_negative=True),
            ItemBias(non_negative=True),
            TimeBias(non_negative=True),
            SplineUserPreferenceShiftFactors(non_negative=True),
        ]
        super().__init__(components=components, **kwargs)



class TimeSVD_NoItemShift(Model):
    """Time SVD algorithm without item shift
    - L. Xiang and Q. Yang, “Time-Dependent Models in Collaborative Filtering Based Recommender System,” in 2009
      IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology, Sep. 2009,
      pp. 450-457. doi: 10.1109/WI-IAT.2009.78.
    """

    def __init__(self, **kwargs):
        components = [
            GlobalAverage(),
            UserBias(),
            ItemBias(),
            TimeBias(),
            StandardFactors(),
            FactorUserBiasShift(),
            FactorPreferenceShift(),
        ]
        super().__init__(components=components, **kwargs)


class SVD_UserItemBias_LinearUserPreferenceShift(Model):
    """Derivative of Time SVD++ algorithm without implicit feedback and item bins
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, **kwargs):
        components = [
            GlobalAverage(),
            UserBias(),
            ItemBias(),
            LinearUserPreferenceShiftFactors(),
        ]
        super().__init__(components=components, **kwargs)

class SVD_UserItemBias_SplineUserPreferenceShift(Model):
    """Derivative of Time SVD++ algorithm without implicit feedback and item bins
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, **kwargs):
        components = [
            GlobalAverage(),
            UserBias(),
            ItemBias(),
            SplineUserPreferenceShiftFactors(),
        ]
        super().__init__(components=components, **kwargs)


class SVD_UserItemTimeBias_SplineUserBiasShift_LinearPreferenceShift(Model):
    """Derivative of Time SVD++ algorithm without implicit feedback and item bins
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, **kwargs):
        components = [
            GlobalAverage(),
            UserBias(),
            SplinesUserBiasShift(),
            ItemBias(),
            TimeBias(),
            LinearUserPreferenceShiftFactors(),
        ]
        super().__init__(components=components, **kwargs)


class SVD_UserItemTimeBias_SplineUserBiasShift_SplinePreferenceShift(Model):
    """Derivative of Time SVD++ algorithm without implicit feedback and item bins
    - Y. Koren, “Collaborative filtering with temporal dynamics,” in Proceedings of the 15th ACM SIGKDD international
      conference on Knowledge discovery and data mining, in KDD '09. New York, NY, USA: Association for Computing
      Machinery, Jun. 2009, pp. 447-456. doi: 10.1145/1557019.1557072.
    """

    def __init__(self, **kwargs):
        components = [
            GlobalAverage(),
            UserBias(),
            SplinesUserBiasShift(),
            ItemBias(),
            TimeBias(),
            SplineUserPreferenceShiftFactors(),
        ]
        super().__init__(components=components, **kwargs)

class SVD_UserItemTimeBias_SplineUserBiasShift_FactorPreferenceShift(Model):
    """Derivative of Time SVD algorithm without without item shift and with spline bias shift instead of factor bias shift
    - L. Xiang and Q. Yang, “Time-Dependent Models in Collaborative Filtering Based Recommender System,” in 2009
      IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology, Sep. 2009,
      pp. 450-457. doi: 10.1109/WI-IAT.2009.78.
    """

    def __init__(self, **kwargs):
        components = [
            GlobalAverage(),
            UserBias(),
            SplinesUserBiasShift(),
            ItemBias(),
            TimeBias(),
            StandardFactors(),
            FactorPreferenceShift(),
        ]
        super().__init__(components=components, **kwargs)