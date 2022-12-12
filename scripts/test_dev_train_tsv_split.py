import pandas as pd
import numpy as np
import os
import sys


def dev_test_train_split(filename):
    """splits tsv file into dev/test/train files 10/10/80 split while
    maintaining original distribution of labels"""
    assert os.path.exists(filename)
    # import dataset
    df = pd.read_csv(filename, sep="\t")
    # set constant variables
    res = []  # for returning result
    np.random.seed(42)  # for reproducibility
    SIZE = df.shape[0] // 10 // len(df.word.unique())  # n of tweets/expression
    # create dictionary containing the proportion of
    # dogwhistle/conventional for each expression
    proportions = {}
    for i in df.word.unique():
        dg = df.loc[(df["word"] == i) & (df["label"] == 1)]
        c = df.loc[(df["word"] == i) & (df["label"] == 0)]
        proportions[i] = len(dg) / (len(dg) + len(c))
    for current_set in ["dev", "test"]:
        # initialize empty dataframe for current set
        df_subset = pd.DataFrame({"id": int(), "text": str(), "label": int()}, index=[])
        for i in df.word.unique():
            # how many expressions to extract respective its type, rounded down
            dogwhistle_rel = int(SIZE * proportions[i])
            conventional_rel = SIZE - dogwhistle_rel
            # these need to be defined again as df changes for every iteration
            dg = df.loc[(df["word"] == i) & (df["label"] == 1)]
            c = df.loc[(df["word"] == i) & (df["label"] == 0)]
            # using proportion, extract indices for respective type
            dg_idx = np.random.choice(dg.index, size=dogwhistle_rel, replace=False)
            c_idx = np.random.choice(c.index, size=conventional_rel, replace=False)
            drop_indices = np.append(dg_idx, c_idx)
            # append sample to subset and remove from original dataframe
            rows = df.loc[drop_indices]
            df = df.drop(drop_indices, axis=0)
            df_subset = pd.concat((df_subset, rows))
        res.append(df_subset.text.values)
        res.append(df_subset.label.values)
    res.append(df.text.values)
    res.append(df.label.values)
    return tuple(res)


if __name__ == "__main__":
    assert os.path.exists(sys.argv[1])
    for i in ["train", "test", "dev"]:
        assert not os.path.exists(f"{sys.argv[1][:-4]}_{i}")

    X_dev, y_dev, X_test, y_test, X_train, y_train = dev_test_train_split(sys.argv[1])
    for i in ["train", "test", "dev"]:
        df = pd.DataFrame(
            {"text": eval(f"X_{i}"), "label": eval(f"y_{i}")}, columns=["text", "label"]
        )
        df.to_csv(f"{sys.argv[1][:-4]}_{i}.tsv", sep="\t", index=False)
