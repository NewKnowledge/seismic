import numpy as np
from scipy.special import chdtri

from seismic_utils import integral_memory_kernel, memory_ccdf


def get_infectiousness(
    share_time, degree, p_time, max_window=7200, min_window=300, min_count=5
):
    """Estimate the infectiousness of an information cascade

    Arguments:
        share_time {numpy.ndarray} -- observed resharing times, sorted, share_time[0]=0
        degree {numpy.ndarray} -- observed node degrees
        p_time {numpy.ndarray} -- equally spaced vector of time to estimate the infectiousness, p_time[0]=0

    Keyword Arguments:
        max_window {int} -- maximum span of the locally weight kernel (default: {7200})
        min_window {int} -- minimum span of the locally weight kernel (default: {300})
        min_count {int} -- the minimum number of resharings included in the window (default: {5})

    Returns:
        tuple -- returns the tuple of vectors (infectiousness, p_up, p_low). The vectors represent the infectiousness
        and its upper and lower bounds for each time in the given `p_time`
    """
    share_time = np.sort(share_time)
    slopes = 2 / (p_time + 1e-8)
    slopes[slopes < 1 / max_window] = 1 / max_window
    slopes[slopes > 1 / min_window] = 1 / min_window

    windows = (p_time + 1e-8) / 2
    windows[windows > max_window] = max_window
    windows[windows < min_window] = min_window

    for j in range(len(p_time)):
        ind = (share_time >= p_time[j] - windows[j]) & (share_time < p_time[j])

        if len(ind) < min_count:
            ind2 = share_time < p_time[j]
            lcv = len(ind2)
            ind = ind2[max(lcv - min_count, 0) : lcv]
            slopes[j] = 1 / (p_time[j] - share_time[ind[1]])
            windows[j] = p_time[j] - share_time[ind[1]]

    M_I = np.zeros((len(share_time), len(p_time)))
    for j in range(len(p_time)):
        M_I[:, j] = degree * integral_memory_kernel(
            p_time[j], share_time, slopes[j], windows[j]
        )

    infectiousness_seq = np.zeros(len(p_time))
    p_low_seq = np.zeros(len(p_time))
    p_up_seq = np.zeros(len(p_time))
    share_time = share_time[1:]

    for j in range(len(p_time)):
        share_time_tri = share_time[
            (share_time >= p_time[j] - windows[j]) & (share_time < p_time[j])
        ]
        rt_count_weighted = np.sum(slopes[j] * (share_time_tri - p_time[j]) + 1)
        I = np.sum(M_I[:, j])
        rt_num = len(share_time_tri)
        if rt_count_weighted == 0:
            continue
        else:
            infectiousness_seq[j] = rt_count_weighted / I
            quant_low = chdtri(
                2 * rt_num, 0.95
            )  # this function is weird and you hand it 1-quantile
            p_low_seq[j] = infectiousness_seq[j] * quant_low / (2 * rt_num)

            quant_up = chdtri(2 * rt_num, 0.05)
            p_up_seq[j] = infectiousness_seq[j] * quant_up / (2 * rt_num)

    return infectiousness_seq, p_up_seq, p_low_seq


def pred_cascade(p_time, infectiousness, share_time, degree, n_star=100):
    """Predict the popularity of information cascade

    Arguments:
        p_time {numpy.ndarray} -- equally spaced vector of time to estimate the total number of retweets, p_time[0]=0
        infectiousness {numpy.ndarray} -- a vector of estimated infectiousness, returned by `get_infectiousness`
        share_time {numpy.ndarray} -- observed resharing times, sorted, share_time[0]=0
        degree {numpy.ndarray} -- observed node degrees

    Keyword Arguments:
        n_star {int or numpy.ndarray} -- the average node degree in the social network (default: {100})
        features_return {bool} -- [description] (default: {False})

    Returns:
        numpy.ndarray -- a vector of predicted populatiry at each time in `p_time`
    """
    # n_star should a vector of the same length as p_time
    if not isinstance(n_star, np.ndarray):
        n_star = np.ones(len(p_time)) * n_star

    prediction = np.zeros(len(p_time))
    for i in range(len(p_time)):
        share_time_now = share_time[share_time <= p_time[i]]
        nf_now = degree[share_time <= p_time[i]]
        rt0 = np.sum(share_time <= p_time[i]) - 1
        rt1 = np.sum(
            nf_now * infectiousness[i] * memory_ccdf(p_time[i] - share_time_now)
        )
        prediction[i] = rt0 + rt1 / (1 - infectiousness[i] * n_star[i])
        if infectiousness[i] > 1 / n_star[i]:
            prediction[i] = np.inf

    return prediction
