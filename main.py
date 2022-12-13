from activebert import *
from preprocessing import DataLoader
import argparse
import pickle
import os
import sys
import io


class CPU_Unpickler(pickle.Unpickler):
    """unpickles model trained using CUDA with CPU instead
    https://stackoverflow.com/questions/57081727"""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def main(*argv, predict=False):
    if predict is True:
        with open(argv[0], "rb") as f:
            print("Importing model for prediction")
            al = CPU_Unpickler(f).load()
        try:
            print("Model imported. Enter query below:")
            while True:
                x = input("> ")
                print(f"\t{'🐕 Doggie' if al.predict([x])[0] == 1 else '🐈 Non-doggie'}")
        except KeyboardInterrupt:
            sys.exit(1)
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
    # In order to import models created using previous versions of the code
    # (where there were no classes). This is only used if one uses a model for
    # predicting.
    def lr_schedule(current_step):
        return 0

    parser = argparse.ArgumentParser(
        description="finetune huggingface model for dogwhistle identification"
    )
    parser.add_argument(
        "--predict",
        metavar="PKL_MODEL",
        dest="args_predict",
        required=False,
        default=None,
        help="Predict using existing model (PKL file)",
    )
    parser.add_argument(
        "-train",
        metavar="FILE",
        type=str,
        dest="train",
        required=False,
        default=None,
        help="tsv file containing training set",
    )
    parser.add_argument(
        "-dev",
        metavar="FILE",
        type=str,
        dest="dev",
        required=False,
        default=None,
        help="tsv file containing dev set",
    )
    parser.add_argument(
        "-test",
        metavar="FILE",
        type=str,
        dest="test",
        required=False,
        default=None,
        help="tsv file containing test set",
    )
    parser.add_argument(
        "-pool",
        metavar="FILE",
        type=str,
        dest="unannotated",
        required=False,
        default=None,
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
    if args.train is None or args.dev is None or args.test is None or args.pool is None:
        if args.args_predict is not None:
            assert os.path.exists(args.args_predict)
            main(args.args_predict, predict=True)
        else:
            print("ERROR: Can't perform active learning without providing datasets!")
            sys.exit(1)
    main(args.train, args.dev, args.test, args.unannotated, args.unseen)
