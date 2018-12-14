import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from seismic import get_infectiousness, pred_cascade


def plot_cascade(share_time, p_time, pred_total, smooth_pred, fname=f"prediction.jpg"):
    plt.clf()
    true_total = np.ones(len(pred_total)) * len(share_time)
    plt.plot(p_time, np.vstack([pred_total, smooth_pred, true_total]).T)

    # plot cumulative retweet count
    plt.plot(share_time[1:], np.arange(len(share_time))[1:])

    plt.xlabel("time (s)")
    plt.ylabel("post count")
    plt.legend(["prediction", "smoothed", "total", "cumulative"])
    plt.savefig(os.path.join("plots", fname))


def main(num_preds=100, num_inf_preds=1000):

    print("reading index csv")
    index_df = pd.read_csv(
        "data/index.csv", dtype={"start_ind": np.int32, "end_ind": np.int32}
    )

    # keep top num_pred tweet cascades by total count
    index_df["num_total"] = index_df["end_ind"] - index_df["start_ind"]
    index_df = index_df.sort_values("num_total", ascending=False)
    index_df = index_df.iloc[:num_preds]

    print("reading data csv")
    data_df = pd.read_csv("data/data.csv")

    # for row in index_df.itertuples(index=False):
    for i, row in index_df.iterrows():
        print("cascade", i)

        # cascade_df = data_df.iloc[row[2]-1:row[3]-1]
        cascade_df = data_df.iloc[int(row["start_ind"]) - 1 : int(row["end_ind"]) - 1]
        max_time = np.max(cascade_df["relative_time_second"])

        # TODO start from 0 or from time of first retweet?
        p_time = np.linspace(
            cascade_df["relative_time_second"].iloc[1], max_time, num=num_inf_preds
        )
        share_time = cascade_df["relative_time_second"].values
        degree = cascade_df["number_of_followers"].values

        infectiousness, p_up, p_low = get_infectiousness(share_time, degree, p_time)
        pred_total = pred_cascade(p_time, infectiousness, share_time, degree)
        # smooth using 2 day window to remove some of the daily periodic trends
        window = np.ceil(num_inf_preds / max_time * 86400 * 2).astype(int)
        if window % 2 == 0:
            window += 1
        smooth_pred = savgol_filter(pred_total, window, 3)
        # pred_up = pred_cascade(p_time, p_up, share_time, degree)
        # pred_low = pred_cascade(p_time, p_low, share_time, degree)

        plot_cascade(
            share_time, p_time, pred_total, smooth_pred, fname=f"prediction{i}.jpg"
        )


if __name__ == "__main__":
    main(num_preds=4)
