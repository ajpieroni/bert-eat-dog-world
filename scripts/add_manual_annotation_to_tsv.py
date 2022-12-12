import pandas as pd
import numpy as np
import json
import argparse


def main(tsv_filename, annotation_json, pool_filename):
    # import files given as input
    with open(annotation_json, "r") as f:
        annotations = json.load(f)
    df = pd.read_csv(tsv_filename, sep="\t")
    al = pd.read_csv(pool_filename, sep="\t")

    # extract rows containing annotations from pool
    rows = al.loc[al["id"].isin([int(x) for x in annotations.keys()])]
    for i in [int(x) for x in annotations.keys()]:
        rows.loc[rows["id"] == i, ["label"]] = annotations[str(i)]
    rows = rows.drop(["id", "word"], axis=1)

    # edit pool dataframe and export
    al = al.drop(rows.index, axis=0)
    al.to_csv(pool_filename, sep="\t", index=False)

    # concatenate tsv of train/test/dev with newly extracted rows
    df = pd.concat([df, rows], ignore_index=True)
    df.to_csv(tsv_filename, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="add json file containing manual annotations to tsv file"
    )
    parser.add_argument(
        "-j",
        metavar="FILE",
        type=str,
        dest="json_file",
        required=True,
        help="json file containing manual annotations",
    )
    parser.add_argument(
        "-p",
        metavar="FILE",
        type=str,
        dest="pool",
        required=True,
        help="tsv file containing unannotated data",
    )
    parser.add_argument(
        "-set",
        metavar="FILE",
        type=str,
        dest="set_file",
        required=False,
        default=None,
        help="tsv file containing train/test/dev-set",
    )
    args = parser.parse_args()
    main(args.set_file, args.json_file, args.pool)
