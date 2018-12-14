import numpy as np


def memory_pdf(t: float, theta=0.2314843, cutoff=300, c=0.0006265725):
    """Probability density function for reaction time (aka memory kernel)

    Arguments:
        t {float} -- time

    Keyword Arguments:
        theta {float} -- exponent of the power law (default: {0.2314843})
        cutoff {float} -- the cutoff value where the density changes from constant to power law (default: {300})
        c {float} -- c the constant density when t is less than the cutoff (default: {0.0006265725})

    Returns:
        float -- returns the density function at t
    """
    if t < cutoff:
        return c
    else:
        return c * np.exp((np.log(t) - np.log(cutoff)) * (-(1 + theta)))


def memory_ccdf(t: float, theta=0.2314843, cutoff=300, c=0.0006265725):
    """cumulative distribution function for reaction time (aka memory kernel)

    Arguments:
        t {float} -- time

    Keyword Arguments:
        theta {float} -- exponent of the power law (default: {0.2314843})
        cutoff {float} -- the cutoff value where the density changes from constant to power law (default: {300})
        c {float} -- c the constant density when t is less than the cutoff (default: {0.0006265725})

    Returns:
        float -- returns the cumulative distribution function at t
    """
    t[t < 0] = 0
    ccdf = np.zeros(len(t))
    index1 = t <= cutoff
    index2 = t > cutoff
    ccdf[index1] = 1 - c * t[index1]
    ccdf[index2] = c * cutoff ** (1 + theta) / theta * (t[index2] ** (-theta))
    return ccdf


def linear_kernel(t1, t2, ptime, slope, c=0.0006265725):
    """Integral of linear kernel

    Arguments:
        t1 {numpy.ndarray} -- a vector of integral lower limit
        t2 {numpy.ndarray} -- a vector of integral upper limit
        ptime {float} -- the time to estimate infectiousness and predict for popularity
        slope {float} -- [description]

    Keyword Arguments:
        c {float} -- c the constant density when t is less than the cutoff (default: {0.0006265725})

    Returns:
        numpy.ndarray -- integral from vector t1 to vector t2 of c*[slope(t-ptime) + 1];
    """
    # indefinite integral is c*(t-ptime*slope*t+(slope*t^2)/2)
    return c * (t2 - ptime * slope * t2 + slope * t2 ** 2 / 2) - c * (
        t1 - ptime * slope * t1 + slope * t1 ** 2 / 2
    )

def power_kernel(
    t1, t2, ptime, share_time, slope, theta=0.2314843, cutoff=300, c=0.0006265725
):
    """Integral of power law kernel

    Arguments:
        t1 {numpy.ndarray} -- a vector of integral lower limit
        t2 {numpy.ndarray} -- a vector of integral upper limit
        ptime {float} -- the time to estimate infectiousness and predict for popularity
        share_time {numpy.ndarray} -- observed resharing times, sorted, share_time[0]=0
        slope {float} -- slope of the linear kernel

    Keyword Arguments:
        theta {float} -- exponent of the power law (default: {0.2314843})
        cutoff {float} -- the cutoff value where the density changes from constant to power law (default: {300})
        c {float} -- c the constant density when t is less than the cutoff (default: {0.0006265725})

    Returns:
        numpy.ndarray -- integral from vector t1 to vector t2 of power law kernel
    """
    # black makes this part hard to read, may need to break up into components
    return c * cutoff**(1 + theta) * (t2 - share_time)**(-theta) \
        * (share_time * slope-theta + (theta - 1) * ptime * slope - theta * slope * t2 + 1) / ((theta - 1) * theta) \
        - c * cutoff**(1 + theta) * (t1 - share_time)**(-theta) \
        * (share_time * slope - theta + (theta - 1) * ptime * slope-theta * slope * t1 + 1) / ((theta - 1) * theta)

def integral_memory_kernel(
    p_time, share_time, slope, window, theta=0.2314843, cutoff=300, c=0.0006265725
):
    """Integration with respect to locally weighted kernel

    Arguments:
        p_time {float} -- the time to estimate infectiousness and predict for popularity
        share_time {numpy.ndarray} -- observed resharing times, sorted, share_time[0]=0
        slope {float} -- slope of the linear kernel
        window {float} -- size of the linear kernel

    Keyword Arguments:
        theta {float} -- exponent of the power law (default: {0.2314843})
        cutoff {float} -- the cutoff value where the density changes from constant to power law (default: {300})
        c {float} -- c the constant density when t is less than the cutoff (default: {0.0006265725})

    Returns:
        numpy.ndarray -- value of integral of the memory kernel evaluated at times in p_time
    """
    # TODO why are theta and c passed in and not used?
    index1 = p_time <= share_time
    index2 = (p_time > share_time) & (p_time <= share_time + cutoff)
    index3 = (p_time > share_time + cutoff) & (p_time <= share_time + window)
    index4 = (p_time > share_time + window) & (p_time <= share_time + window + cutoff)
    index5 = p_time > share_time + window + cutoff
    integral = np.array([np.nan] * len(share_time))
    integral[index1] = 0
    integral[index2] = linear_kernel(share_time[index2], p_time, p_time, slope)
    integral[index3] = linear_kernel(
        share_time[index3], share_time[index3] + cutoff, p_time, slope
    ) + power_kernel(
        share_time[index3] + cutoff, p_time, p_time, share_time[index3], slope
    )
    integral[index4] = linear_kernel(
        p_time - window, share_time[index4] + cutoff, p_time, slope
    ) + power_kernel(
        share_time[index4] + cutoff, p_time, p_time, share_time[index4], slope
    )
    integral[index5] = power_kernel(
        p_time - window, p_time, p_time, share_time[index5], slope
    )
    return integral
