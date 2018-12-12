import numpy as np

#' Memory kernel
#'
#' Probability density function and complementary cumulative distribution function
#' for the human reaction time.
#' @keywords internal
#'
#' @param t time
#' @param theta exponent of the power law
#' @param cutoff the cutoff value where the density changes from constant to power law
#' @param c the constant density when t is less than the cutoff
#' @return the density at t
#' @details default values are measured from a real Twitter data set.
#' @return \code{memory.pdf} returns the density function at t.
#' \code{memory.ccdf} returns the ccdf (probabilty of greater than t).
#'


def memory_pdf(t: float, theta=0.2314843, cutoff=300, c=0.0006265725):
    if t < cutoff:
        return c
    else:
       #return(c*   exp((   log(t) -    log(cutoff))*(-(1+theta))))
       return  c*np.exp((np.log(t) - np.log(cutoff))*(-(1+theta)))


def memory_ccdf(t, theta=0.2314843, cutoff=300, c=0.0006265725):
#   t[t<0] <- 0
    t[t<0] = 0
#   ccdf <- rep(0, length(t))
    ccdf = np.zeros(len(t))
#   index1 <- which(t <= cutoff)
    index1 = t <= cutoff
#   index2 <- which(t > cutoff)
    index2 = t > cutoff
#   ccdf[index1] <- 1 - c*t[index1]
    ccdf[index1] =  1 - c*t[index1]
#   ccdf[index2] <-c*cutoff ^(1+theta)/theta*(t[index2] ^(-theta))
    ccdf[index2] = c*cutoff**(1+theta)/theta*(t[index2]**(-theta))
#   ccdf
    return ccdf



#' Integration with respect to locally weighted kernel
#'
#' @keywords internal
#'
#' @param t1 a vector of integral lower limit
#' @param t2 a vector of integral upper limit
#' @param ptime the time (a scalar) to estimate infectiousness and predict for popularity
#' @param slope slope of the linear kernel
#' @param window size of the linear kernel
#' @inheritParams memory.pdf
#' @inheritParams get.infectiousness
#' @return \code{linear.kernel} returns the integral from vector t1 to vector t2 of
#' c*[slope(t-ptime) + 1];
#' \code{power.kernel} returns the integral from vector t1 to vector 2 of c*((t-share.time)/cutoff)^(-(1+theta))[slope(t-ptime) + 1];
#' \code{integral.memory.kernel} returns the vector with ith entry being integral_-inf^inf phi_share.time[i]*kernel(t-p.time)
def linear_kernel(t1, t2, ptime, slope, c=0.0006265725):
    # indefinite integral is c*(t-ptime*slope*t+(slope*t^2)/2)
#   return(c*(t2-ptime*slope*t2+slope*t2 ^2/2) - c*(t1-ptime*slope*t1+slope*t1 ^2/2))
    return(c*(t2-ptime*slope*t2+slope*t2**2/2) - c*(t1-ptime*slope*t1+slope*t1**2/2))


def power_kernel(
    t1, t2, ptime, share_time, slope, theta=0.2314843, cutoff=300, c=0.0006265725
):
#   return (c*cutoff^(1+theta)*(t2-share.time)  ^(-theta)*(share.time*slope-theta+(theta-1)*ptime*slope-theta*slope*t2+1)/((theta-1)*theta) - c*cutoff ^(1+theta)*(t1-share.time) ^(-theta)*(share.time*slope-theta+(theta-1)*ptime*slope-theta*slope*t1+1)/((theta-1)*theta))
    return (c*cutoff**(1+theta)*(t2-share_time)**(-theta)*(share_time*slope-theta+(theta-1)*ptime*slope-theta*slope*t2+1)/((theta-1)*theta) - c*cutoff**(1+theta)*(t1-share_time)**(-theta)*(share_time*slope-theta+(theta-1)*ptime*slope-theta*slope*t1+1)/((theta-1)*theta))

#' @describeIn linear.kernel
def integral_memory_kernel(p_time, share_time, slope, window, theta=0.2314843, cutoff=300, c=0.0006265725):
    # TODO why is theta and c passed in and not used?
    # index1 <- which(p.time <= share.time)
    index1 = p_time <= share_time
    # index2 <- which(p.time > share.time & p.time <= share.time + cutoff)
    index2 = (p_time > share_time) & (p_time <= share_time + cutoff)
    # index3 <- which(p.time > share.time + cutoff & p.time <= share.time + window)
    index3 = (p_time > share_time + cutoff) & (p_time <= share_time + window)
    # index4 <- which(p.time > share.time + window & p.time <= share.time + window + cutoff)
    index4 = (p_time > share_time + window) & (p_time <= share_time + window + cutoff)
    # index5 <- which(p.time > share.time + window + cutoff)
    index5 = p_time > share_time + window + cutoff
    # integral <- rep(NA, length(share.time))
    integral = np.array([np.nan]*len(share_time))
    # integral[index1] <- 0
    integral[index1] = 0
    # integral[index2] <- linear.kernel(share.time[index2], p.time, p.time, slope)
    integral[index2] = linear_kernel(share_time[index2], p_time, p_time, slope)
    # integral[index3] <- linear.kernel(share.time[index3], share.time[index3] + cutoff, p.time, slope) + power.kernel(share.time[index3]+cutoff, p.time, p.time, share.time[index3], slope)
    integral[index3] = linear_kernel(share_time[index3], share_time[index3] + cutoff, p_time, slope) + power_kernel(share_time[index3]+cutoff, p_time, p_time, share_time[index3], slope)
    # integral[index4] <- linear.kernel(p.time-window, share.time[index4]+cutoff, p.time, slope) + power.kernel(share.time[index4]+cutoff, p.time, p.time, share.time[index4], slope)
    integral[index4] = linear_kernel(p_time-window, share_time[index4]+cutoff, p_time, slope) + power_kernel(share_time[index4]+cutoff, p_time, p_time, share_time[index4], slope)
    # integral[index5] <- power_kernel(p.time-window, p.time, p.time, share.time[index5], slope)
    integral[index5] = power_kernel(p_time-window, p_time, p_time, share_time[index5], slope)
    # return(integral)
    return integral
