import pandas as pd
import numpy as np
import os
import sys

def dev_test_train_split(filename):
    assert os.path.exists(filename)
    
    df = pd.read_csv(filename, sep="\t")
    
    res = [] 
    np.random.seed(42)
    SIZE = df.shape[0] // 10
    
    for current_set in ["dev", "test"]:
        df_subset = pd.DataFrame(columns=['text', 'label'])
        for label in df.label.unique():
            df_label = df[df['label'] == label]
            subset_size = SIZE // len(df.label.unique())
            subset = df_label.sample(n=subset_size, replace=False)
            df_subset = pd.concat([df_subset, subset])
            df = df.drop(subset.index)
        res.extend([df_subset.text.values, df_subset.label.values])
    
    res.extend([df.text.values, df.label.values])
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
