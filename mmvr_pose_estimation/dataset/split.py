from argparse import ArgumentParser

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split


def split(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    kp: npt.ArrayLike,
    *,
    test_size: float = 0.125,
    train_output: str = "train.npz",
    test_output: str = "test.npz",
):
    X_train, X_test, y_train, y_test, kp_train, kp_test = train_test_split(
        X, y, kp, test_size=test_size, shuffle=True
    )

    np.savez(train_output, X=X_train, y=y_train, kp=kp_train)
    np.savez(test_output, X=X_test, y=y_test, kp=kp_test)


def main():
    parser = ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--test-size", default=0.125, type=float, dest="test_size")
    parser.add_argument("--train-output", dest="train_npz")
    parser.add_argument("--test-output", dest="test_npz")
    args = parser.parse_args()

    dataset = np.load(args.file)
    X, y, kp = dataset["X"], dataset["y"], dataset["kp"]
    split(X, y, kp, test_size=args.test_size, train_output=args.train_npz, test_output=args.test_npz)

if __name__ == "__main__":
    main()
