import numpy as np


def calc_spline_function(user, t_ui, spline_factors, spline_kernels, gamma, is_for_factors=False):
    """Returns the spline function value for a given user and time as
    well as the first derivate for all kernels.
    """
    decay_factor = np.exp(-gamma * np.abs(t_ui - spline_kernels[user]))
    denominator = np.sum(decay_factor)
    numerator = np.sum(decay_factor * spline_factors[user], axis=1 if is_for_factors else 0)
    return (numerator / denominator, decay_factor / denominator)


def calc_dev(t_ui, t_u_mean, beta):
    diff = t_ui - t_u_mean
    return np.sign(diff) * np.abs(diff) ** beta


def clip_gradient(gradient, c=1):
    """Clip the gradient to mitigate the exploding gradient problem."""
    norm = np.linalg.norm(gradient)
    if norm > c:
        gradient = c * (gradient / norm)
    return gradient


def pandas_to_numpy(
    students, exercises, pd_known_ratings, pd_t_u=None, pd_t_i=None, pd_t_u_mean=None, pd_t_u_max=None, pd_n_ratings_u=None
):
    """Convert pandas dataframe to numpy array and map users and items to ids."""
    pd_known_ratings = pd_known_ratings.copy()

    # create mapping for users and items to ids
    user_to_id = {user: i for i, user in enumerate(students)}
    item_to_id = {item: i for i, item in enumerate(exercises)}
    f_user_to_id = lambda user: user_to_id[user]
    f_item_to_id = lambda item: item_to_id[item]

    # map ratings
    pd_known_ratings["student"] = pd_known_ratings["student"].apply(f_user_to_id)
    pd_known_ratings["exercise"] = pd_known_ratings["exercise"].apply(f_item_to_id)
    know_ratings = pd_known_ratings.to_numpy()

    if pd_t_u is not None and pd_t_i is not None:
        pd_t_u = pd_t_u[pd_t_u.index.isin(students)].copy()
        pd_t_i = pd_t_i[pd_t_i.index.isin(exercises)].copy()

        # map t_u and t_i
        pd_t_u.index = pd_t_u.index.map(f_user_to_id)
        pd_t_u = pd_t_u.sort_index()
        pd_t_i.index = pd_t_i.index.map(f_item_to_id)
        pd_t_i = pd_t_i.sort_index()
        # convert to numpy
        np_t_u = pd_t_u.to_numpy()
        np_t_i = pd_t_i.to_numpy()

        return know_ratings, np_t_u, np_t_i, user_to_id, item_to_id
    elif pd_t_u_mean is not None:
        pd_t_u_mean = pd_t_u_mean[pd_t_u_mean.index.isin(students)].copy()

        # map t_u_mean
        pd_t_u_mean.index = pd_t_u_mean.index.map(f_user_to_id)
        pd_t_u_mean = pd_t_u_mean.sort_index()
        # convert to numpy
        np_t_u_mean = pd_t_u_mean.to_numpy()

        return know_ratings, np_t_u_mean, user_to_id, item_to_id
    elif pd_t_u_max is not None:
        pd_t_u = pd_t_u[pd_t_u.index.isin(students)].copy()
        pd_t_u_max = pd_t_u_max[pd_t_u_max.index.isin(students)].copy()
        pd_n_ratings_u = pd_n_ratings_u[pd_n_ratings_u.index.isin(students)].copy()

        # map
        pd_t_u.index = pd_t_u.index.map(f_user_to_id)
        pd_t_u = pd_t_u.sort_index()
        pd_t_u_max.index = pd_t_u_max.index.map(f_user_to_id)
        pd_t_u_max = pd_t_u_max.sort_index()
        pd_n_ratings_u.index = pd_n_ratings_u.index.map(f_user_to_id)
        pd_n_ratings_u = pd_n_ratings_u.sort_index()
        # convert to numpy
        np_t_u = pd_t_u.to_numpy()
        np_t_u_max = pd_t_u_max.to_numpy()
        np_n_ratings_u = pd_n_ratings_u.to_numpy()

        return know_ratings, np_t_u, np_t_u_max, np_n_ratings_u, user_to_id, item_to_id
    else:
        return know_ratings, user_to_id, item_to_id
