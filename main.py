from activebert import ActiveBert
from preprocessing import DataLoader
import argparse
import pickle
import os
import sys


def main(*argv):
    if os.path.exists("model.pkl"):
        print("Existing model found. Importing...")
        with open("model.pkl", "rb") as f:
            al = pickle.load(f)
    else:
        al = ActiveBert(DataLoader(argv))
    try:
        al.query_loop()
    except KeyboardInterrupt:
        print("\nUser interrupt.")
        if al.dl.X_unseen is not None and al.dl.y_unseen is not None:
            unseen_score = al.calc_score(al.dl.X_unseen, al.dl.y_unseen)
            print(f"Accuracy of model of unseen tweets: {unseen_score}")
        final_score = al.calc_score(al.dl.X_test, al.dl.y_test)
        print(f"Accuracy of model (test set): {final_score}")
        save_model(al)
        print("Model was saved as model.pkl\nExiting...")
        sys.exit(0)


def save_model(model):
    if os.path.exists("model.pkl"):
        while True:
            ans = input("learner.pkl already exists. Overwrite? y/n\t")
            if ans == "n":
                print("Model was not saved. Exiting...")
                sys.exit(0)
            elif ans == "y":
                break
    with open("model.pkl", "wb") as export:
        pickle.dump(model, export)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="finetune huggingface model for dogwhistle identification"
    )
    parser.add_argument(
        "-train",
        metavar="FILE",
        type=str,
        dest="train",
        required=True,
        help="tsv file containing training set",
    )
    parser.add_argument(
        "-dev",
        metavar="FILE",
        type=str,
        dest="dev",
        required=True,
        help="tsv file containing dev set",
    )
    parser.add_argument(
        "-test",
        metavar="FILE",
        type=str,
        dest="test",
        required=True,
        help="tsv file containing test set",
    )
    parser.add_argument(
        "-pool",
        metavar="FILE",
        type=str,
        dest="unannotated",
        required=True,
        help="tsv file containing unannotated data",
    )
    parser.add_argument(
        "-unseen",
        metavar="FILE",
        type=str,
        dest="unseen",
        required=False,
        default=None,
        help="tsv file containing unseen data",
    )
    args = parser.parse_args()
    main(args.train, args.dev, args.test, args.unannotated, args.unseen)
