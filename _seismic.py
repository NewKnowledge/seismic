import numpy as np
from scipy.special import chdtri
from seismic_utils import integral_memory_kernel, memory_ccdf
import pandas as pd

#' Estimate the infectiousness of an information cascade
#'
#' @param share_time observed resharing times, sorted, share_time[1] =0
#' @param degree observed node degrees
#' @param p_time equally spaced vector of time to estimate the infectiousness, p_time[1]=0
#' @param max_window maximum span of the locally weight kernel
#' @param min_window minimum span of the locally weight kernel
#' @param min_count the minimum number of resharings included in the window
#' @details Use a triangular kernel with shape changing over time. At time p_time, use a triangluer kernel with slope = min(max(1/(\code{p_time}/2), 1/\code{min_window}), \code{max_window}).
#' @return a list of three vectors: \itemize{
#' \item infectiousness. the estimated infectiousness
#' \item p.up. the upper 95 percent approximate confidence interval
#' \item p.low. the lower 95 percent approximate confidence interval
#' }
#' @export
#' @examples
#' data(tweet)
#' pred.time <- seq(0, 6 * 60 * 60, by = 60)
#' infectiousness <- get.infectiousness(tweet[, 1], tweet[, 2], pred.time)
#' plot(pred.time, infectiousness$infectiousness)


def get_infectiousness(
    share_time,
    degree,
    p_time,
    max_window = 2 * 60 * 60,
    min_window = 300,
    min_count = 5,
):

    # ix <- sort(share_time, index.return=TRUE)$ix
    # share_time <- share_time[ix]
    share_time = np.sort(share_time)

#   slopes <- 1/(p_time/2)
    slopes = 1/(p_time/2) # = 2/p_time?  TODO above says p_time[1]=0, but if so then we divide by 0 here
#   slopes[slopes < 1/max_window] <- 1/max_window
    slopes[slopes < 1/max_window] = 1/max_window
#   slopes[slopes > 1/min_window] <- 1/min_window
    slopes[slopes > 1/min_window] = 1/min_window

#   windows <- p_time/2
    windows = p_time/2
#   windows[windows > max_window] <- max_window
    windows[windows > max_window] = max_window
#   windows[windows < min_window] <- min_window
    windows[windows < min_window] = min_window

#   for(j in c(1:length(p_time))) {
    for j in range(len(p_time)):  # XXX range ok here?
#       ind <- which(share_time >= p_time[j] - windows[j] & share_time < p_time[j])
        ind = (share_time >= p_time[j] - windows[j]) & (share_time < p_time[j])

#       if(length(ind) < min_count) {
        if len(ind) < min_count:
#           ind2 <- which(share_time < p_time[j])
            ind2 = share_time < p_time[j]
#           lcv <- length(ind2)
            lcv = len(ind2)
#           ind <-ind2[max((lcv-min_count),1):lcv]
            ind = ind2[max(lcv-min_count, 0):lcv]  # XXX indices correct?
#           slopes[j] <-1/(p_time[j] - share_time[ind[1]])
            slopes[j] = 1/(p_time[j] - share_time[ind[1]])

#           windows[j] <- p_time[j] - share_time[ind[1]]
            windows[j] = p_time[j] - share_time[ind[1]]

#   M_I <- matrix(0,nrow=length(share_time),ncol=length(p_time))
    M_I = np.zeros((len(share_time), len(p_time)))
#   for(j in 1:length(p_time)){
    for j in range(len(p_time)):  # XXX range ok?
#       M_I[,j] <- degree*integral_memory_kernel(p_time[j], share_time, slopes[j], windows[j])
        M_I[:,j] = degree*integral_memory_kernel(p_time[j], share_time, slopes[j], windows[j])
#   infectiousness_seq <- rep(0, length(p_time))
    infectiousness_seq = np.zeros(len(p_time))
#   p_low_seq <- rep(0, length(p_time))
    p_low_seq = np.zeros(len(p_time))
#   p_up_seq <- rep(0, length(p_time))
    p_up_seq = np.zeros(len(p_time))
#   share_time <- share_time[-1]          #removes the original tweet from retweet
    share_time = share_time[1:]  # XXX slice correct? #removes the original tweet from retweet
#   for(j in c(1:length(p_time))) {
    for j in range(len(p_time)):
#       share_time_tri <- share_time[which(share_time >= p_time[j] - windows[j] & share_time < p_time[j])]
        share_time_tri = share_time[(share_time >= p_time[j] - windows[j]) & (share_time < p_time[j])]
#       rt_count_weighted <-   sum(slopes[j]*(share_time_tri - p_time[j]) + 1)
        rt_count_weighted = np.sum(slopes[j]*(share_time_tri - p_time[j]) + 1)
#       I <- sum(M_I[,j])
        I = np.sum(M_I[:,j])  # XXX slice correct?
#       rt_num <- length(share_time_tri)
        rt_num = len(share_time_tri)
#       if (rt_count_weighted==0)
        if rt_count_weighted == 0:
#           next
            continue
#       else {
        else:
#           infectiousness_seq[j] <- (rt_count_weighted)/I
            infectiousness_seq[j] = rt_count_weighted/I
#           p_low_seq[j] <- infectiousness_seq[j] * qchisq(0.05, 2*rt_num) / (2*rt_num)
            quant_low = chdtri(2*rt_num, 1-0.05) # XXX quantile equivalent?
            p_low_seq[j] = infectiousness_seq[j] * quant_low / (2*rt_num)

#           p_up_seq[j] <- infectiousness_seq[j] * qchisq(0.95, 2*rt_num) / (2*rt_num)
            quant_up = chdtri(2*rt_num, 1-0.95) # XXX quantile equivalent?
            p_up_seq[j] = infectiousness_seq[j] * quant_up / (2*rt_num)

#   ## p_low_seq[is.nan(p_low_seq)] <- 0
#   ## p_up_seq[is.nan(p_up_seq)] <- 0
#   list(infectiousness = infectiousness_seq, p.up = p_up_seq, p.low = p_low_seq)
    return infectiousness_seq, p_up_seq, p_low_seq


#' Predict the popularity of information cascade
#'
#' @param infectiousness a vector of estimated infectiousness, returned by \code{\link{get.infectiousness}}
#' @param n_star the average node degree in the social network
#' @param features.return if TRUE, returns a matrix of features to be used to further calibrate the prediction
#' @inheritParams get.infectiousness
#' @return a vector of predicted populatiry at each time in \code{p_time}.
#' @export
#' @examples
#' data(tweet)
#' pred.time <- seq(0, 6 * 60 * 60, by = 60)
#' infectiousness <- get.infectiousness(tweet[, 1], tweet[, 2], pred.time)
#' pred <- pred.cascade(pred.time, infectiousness$infectiousness, tweet[, 1], tweet[, 2], n_star = 100)
#' plot(pred.time, pred)

def pred_cascade(p_time, infectiousness, share_time, degree, n_star=100, features_return=False):


#   # n_star should a vector of the same length as p_time
#   if (length(n_star) == 1) {
    if not isinstance(n_star, np.ndarray):
#       n_star <- rep(n_star, length(p_time))
        n_star = np.ones(len(p_time)) * n_star

#   # to train for best n_star, we get feature matrices
#   features <- matrix(0, length(p_time), 3)
    features = np.zeros((len(p_time), 3))

#   prediction <- matrix(0, length(p_time), 1)
    prediction = np.zeros(len(p_time))
#   for (i in 1:length(p_time)) {
    for i in range(len(p_time)):
#       share_time_now <- share_time[share_time <= p_time[i]]
        share_time_now = share_time[share_time <= p_time[i]]
#       nf_now <- degree[share_time <= p_time[i]]
        nf_now = degree[share_time <= p_time[i]]
#       rt0 <- sum(share_time <= p_time[i]) - 1  # XXX does this count the number of elements that satisfy the condition (-1)?
        rt0 = np.sum(share_time <= p_time[i]) - 1
#       rt1 <- sum(nf_now * infectiousness[i] * memory_ccdf(p_time[i] - share_time_now))
        rt1 = np.sum(nf_now * infectiousness[i] * memory_ccdf(p_time[i] - share_time_now))
#       prediction[i] <- rt0 + rt1 / (1 - infectiousness[i]*n_star[i])
        prediction[i] = rt0 + rt1 / (1 - infectiousness[i]*n_star[i])
#       features[i, ] <- c(rt0, rt1, infectiousness[i])
        features[i,:] = (rt0, rt1, infectiousness[i])
#       if (infectiousness[i] > 1/n_star[i]) {
        if infectiousness[i] > 1/n_star[i]:
#           prediction[i] <- Inf
            prediction[i] = np.inf

#   colnames(features) <- c("current_rt", "numerator", "infectiousness")
#   XXX anything to do here? maybe make features a dataframe and name the columns?

#   if (!features_return) {
    if not features_return:
#       prediction
        return prediction
#   } else {
    else:
#       list(prediction = prediction, features = features)
        return prediction, features

def gen_test_data(n_samples=1000, to_file=True):
    share_time = np.random.rand(n_samples) * 10000
    share_time = np.sort(share_time)
    degree = np.random.randint(1, 1000, size=n_samples)
    p_time = np.linspace(0, 10000, num=n_samples)

    df = pd.DataFrame({'share_time': share_time, "degree": degree, 'p_time': p_time})
    if to_file:
        df.to_csv('test-input.csv', index=False)

    return df

def gen_output():
    df = pd.read_csv('test_input.csv')

    infectiousness, p_up, p_low = get_infectiousness(df['share_time'], df['degree'], df['p_time'])
    predicted_total = pred_cascade(df['p_time'], infectiousness, df['share_time'], df['degree'])

    out_df = pd.DataFrame({'infectiousness': infectiousness, 'pred': predicted_total, 'p_up': p_up, 'p_low': p_low})
    out_df.to_csv('py_output.csv')
    # out_df.to_csv('py_output.csv', index=False)

    return out_df

def test_outputs():

    r_df = pd.read_csv('r_output.csv')
    py_df = pd.read_csv('py_output.csv')

    match = np.isclose(r_df['infectiousness'], py_df['infectiousness'])
    mismatch_inds = np.nonzero(~match)[0]
    print('mismatch indices:', mismatch_inds)
    # print(r_df.iloc[mismatch_inds])
    # print(py_df.iloc[mismatch_inds])

    print('number of infectiousness mismatches:', len(r_df) - sum(np.isclose(r_df['infectiousness'], py_df['infectiousness'])))
    # assert np.isclose(r_df['infectiousness'], py_df['infectiousness']).all()

    print('number of pred mismatches:', len(r_df) - sum(np.isclose(r_df['pred'], py_df['pred'])))
    # assert np.isclose(r_df['pred'], py_df['pred']).all()

    print('number of p_up mismatches:', len(r_df) - sum(np.isclose(r_df['p.up'], py_df['p_up'])))
    # assert np.isclose(r_df['p.up'], py_df['p_up']).all()

    print('number of p_low mismatches:', len(r_df) - sum(np.isclose(r_df['p.low'], py_df['p_low'])))
    # assert np.isclose(r_df['p.low'], py_df['p_low']).all()


if __name__ == '__main__':
    # gen_test_data()
    # gen_output()
    test_outputs()
