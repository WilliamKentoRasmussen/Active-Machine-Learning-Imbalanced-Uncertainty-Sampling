import copy
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

import torch
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def load_data(n_init=100):
    training_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=ToTensor()
    )

    indices = []
    threshold = 5
    amount = defaultdict(int)
    targets = training_data.targets

    for digit in range(10):
        digit_indices = (targets == digit).nonzero(as_tuple=True)[0]
        if digit < threshold:
            selected = digit_indices[: len(digit_indices) // 400]   # keep 10%
            amount["below"] += len(selected)
        else:
            selected = digit_indices[: len(digit_indices) // 1000]  # keep 1%
            amount["above"] += len(selected)
        indices.append(selected)

    indices = torch.cat(indices)

    X = training_data.data[indices].float() / 255.0
    y = training_data.targets[indices]

    X_test = test_data.data.float() / 255.0
    y_test = test_data.targets
    X_test= X_test[0:100]
    y_test = y_test[0:100]

    seed = 0
    n = len(indices)
    print(f"Total subset size: {n}")

    sss = ShuffleSplit(n_splits=1, train_size=n_init / n, random_state=seed)
    train_idx, pool_idx = next(sss.split(X.numpy(), y.numpy()))

    data = dict(
        train=dict(
            X=X[train_idx].reshape(len(train_idx), -1).numpy(),
            y=y[train_idx].numpy(),
        ),
        pool=dict(
            X=X[pool_idx].reshape(len(pool_idx), -1).numpy(),
            y=y[pool_idx].numpy(),
        ),
        test=dict(
            X=X_test.reshape(len(X_test), -1).numpy(),
            y=y_test.numpy(),
        ),
    )
    return data


def evaluate_uncertainty(prob, strategy):
    if strategy == "least confident":
        return 1 - prob.max(1)
    elif strategy == "margin":
        ix = np.arange(len(prob))
        p2, p1 = prob.argsort(1)[:, -2:].T
        return 1 - (prob[ix, p1] - prob[ix, p2])
    elif strategy == "entropy":
        # Clip to avoid log(0)
        prob = np.clip(prob, 1e-12, 1.0)
        return -np.sum(prob * np.log2(prob), axis=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def update_data(data, idx):
    """Move the point at pool index `idx` into the training set."""
    data["train"]["X"] = np.append(
        data["train"]["X"], np.atleast_2d(data["pool"]["X"][idx]), axis=0
    )
    data["train"]["y"] = np.append(
        data["train"]["y"], np.atleast_1d(data["pool"]["y"][idx]), axis=0
    )
    data["pool"]["X"] = np.delete(data["pool"]["X"], idx, axis=0)
    data["pool"]["y"] = np.delete(data["pool"]["y"], idx, axis=0)


def fit_model(base_data, paradigm, strategy, n_init, n_iterations, plot=False):
    """
    Parameters
    ----------
    base_data   : the original data dict (never mutated)
    paradigm    : 'active learning' | 'random'
    strategy    : 'least confident' | 'margin' | 'entropy'
    n_init      : initial training-set size (informational only here)
    n_iterations: number of AL/random steps
    plot        : whether to call plot_pool after each step
    """

    data = copy.deepcopy(base_data)

    scores = np.zeros(n_iterations)
    model = LogisticRegression(C=1e1, solver="sag", max_iter=1000)

    for i in range(n_iterations):
        model.fit(data["train"]["X"], data["train"]["y"])

        prob = model.predict_proba(data["pool"]["X"])
        scores[i] = model.score(data["test"]["X"], data["test"]["y"])

  
        if paradigm == "active learning":
            uncertainty = evaluate_uncertainty(prob, strategy)

            idx = int(uncertainty.argmax())
        elif paradigm == "random":
            uncertainty = None
            idx = int(np.random.choice(len(data["pool"]["X"])))
        else:
            raise ValueError(f"Unknown paradigm: {paradigm!r}")

        update_data(data, idx)

    return scores


if __name__ == "__main__":
    N_INIT       = 5
    N_ITERATIONS = 50 - N_INIT
    N_AVG        = 50


    base_data = load_data(n_init=N_INIT)

    print(f"Train  X: {base_data['train']['X'].shape}")
    print(f"Pool   X: {base_data['pool']['X'].shape}")
    print(f"Test   X: {base_data['test']['X'].shape}\n")

    scores_al = np.zeros((N_AVG, N_ITERATIONS))
    scores_rn = np.zeros((N_AVG, N_ITERATIONS))

    for i in tqdm(range(N_AVG), desc="Averaging runs"):
        scores_al[i] = fit_model(base_data, "active learning", "entropy", N_INIT, N_ITERATIONS)
        scores_rn[i] = fit_model(base_data, "random",          "entropy", N_INIT, N_ITERATIONS)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x_axis = np.arange(N_INIT, N_ITERATIONS + N_INIT)
    ax.plot(x_axis, scores_al.mean(0), label="Active learning (entropy)")
    ax.plot(x_axis, scores_rn.mean(0), label="Random")
    ax.fill_between(x_axis,
                    scores_al.mean(0) - scores_al.std(0),
                    scores_al.mean(0) + scores_al.std(0), alpha=0.2)
    ax.fill_between(x_axis,
                    scores_rn.mean(0) - scores_rn.std(0),
                    scores_rn.mean(0) + scores_rn.std(0), alpha=0.2)
    ax.legend()
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Active Learning vs Random Sampling on MNIST")
    fig.tight_layout()
    fig.savefig("./data/sampling.png", dpi=150)
    print("Saved Classification.png")