import numpy as np
import pandas as pd

from seismic import get_infectiousness, pred_cascade


def gen_test_data(n_samples=1000, to_file=True):
    share_time = np.random.rand(n_samples) * 10000
    share_time = np.sort(share_time)
    degree = np.random.randint(1, 1000, size=n_samples)
    p_time = np.linspace(0, 10000, num=n_samples)

    df = pd.DataFrame({"share_time": share_time, "degree": degree, "p_time": p_time})
    if to_file:
        df.to_csv("test-input.csv", index=False)

    return df


def gen_output():
    df = pd.read_csv("test_input.csv")

    infectiousness, p_up, p_low = get_infectiousness(
        df["share_time"], df["degree"], df["p_time"]
    )
    predicted_total = pred_cascade(
        df["p_time"], infectiousness, df["share_time"], df["degree"]
    )

    out_df = pd.DataFrame(
        {
            "infectiousness": infectiousness,
            "pred": predicted_total,
            "p_up": p_up,
            "p_low": p_low,
        }
    )
    out_df.to_csv("py_output.csv")
    # out_df.to_csv('py_output.csv', index=False)

    return out_df


def test_outputs():

    r_df = pd.read_csv("r_output.csv")
    py_df = pd.read_csv("py_output.csv")

    match = np.isclose(r_df["infectiousness"], py_df["infectiousness"])
    mismatch_inds = np.nonzero(~match)[0]
    print("mismatch indices:", mismatch_inds)
    # print(r_df.iloc[mismatch_inds])
    # print(py_df.iloc[mismatch_inds])

    print(
        "number of infectiousness mismatches:",
        len(r_df) - sum(np.isclose(r_df["infectiousness"], py_df["infectiousness"])),
    )
    # assert np.isclose(r_df['infectiousness'], py_df['infectiousness']).all()

    print(
        "number of pred mismatches:",
        len(r_df) - sum(np.isclose(r_df["pred"], py_df["pred"])),
    )
    # assert np.isclose(r_df['pred'], py_df['pred']).all()

    print(
        "number of p_up mismatches:",
        len(r_df) - sum(np.isclose(r_df["p.up"], py_df["p_up"])),
    )
    # assert np.isclose(r_df['p.up'], py_df['p_up']).all()

    print(
        "number of p_low mismatches:",
        len(r_df) - sum(np.isclose(r_df["p.low"], py_df["p_low"])),
    )
    # assert np.isclose(r_df['p.low'], py_df['p_low']).all()


if __name__ == "__main__":
    # gen_test_data()
    # gen_output()
    test_outputs()
