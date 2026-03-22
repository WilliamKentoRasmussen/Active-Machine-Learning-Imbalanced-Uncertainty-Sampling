import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
from collections import defaultdict
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

threshold = 5


def load_data(n_init=20, imbalanced=True, total_per_class=200):
    training_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=ToTensor()
    )

    indices = []
    targets = training_data.targets

    for digit in range(10):
        digit_indices = (targets == digit).nonzero(as_tuple=True)[0]

        # shuffle indices
        perm = torch.randperm(len(digit_indices))
        digit_indices = digit_indices[perm]

        if imbalanced:
            if digit < threshold:
                k = int(0.15 * total_per_class)   # more samples for digits 0–4
            else:
                k = int(0.05 * total_per_class)   # fewer samples for digits 5–9
        else:
            k = int(0.1 * total_per_class)        # equal for all digits

        selected = digit_indices[:k]
        indices.append(selected)

    indices = torch.cat(indices)

    X = training_data.data[indices].float() / 255.0
    y = training_data.targets[indices]

    X_test = test_data.data.float() / 255.0
    y_test = test_data.targets

    X_test = X_test[:1000]
    y_test = y_test[:1000]

    n = len(indices)

    sss = ShuffleSplit(n_splits=1, train_size=n_init, random_state=0)
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


def get_dataset_info(data, dataset_name):
    """Build a DataFrame summarising dataset sizes and class distributions."""
    rows = []
    for split in ("train", "pool", "test"):
        y = data[split]["y"]
        n_total = len(y)
        class_counts = {f"class_{c}": int(np.sum(y == c)) for c in range(10)}
        rows.append({
            "dataset":    dataset_name,
            "split":      split,
            "n_total":    n_total,
            "n_below_threshold": int(np.sum(y < threshold)),   # digits 0–4
            "n_above_threshold": int(np.sum(y >= threshold)),  # digits 5–9
            **class_counts,
        })
    return pd.DataFrame(rows)


def evaluate_uncertainty(prob, strategy):
    if strategy == "least confident":
        return 1 - prob.max(1)
    elif strategy == "margin":
        ix = np.arange(len(prob))
        p2, p1 = prob.argsort(1)[:, -2:].T
        return 1 - (prob[ix, p1] - prob[ix, p2])
    elif strategy == "entropy":
        prob = np.clip(prob, 1e-12, 1.0)
        return -np.sum(prob * np.log2(prob), axis=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def update_data(data, idx, strategy, selected_classes):
    """Move one point from pool → train and record which label was selected."""
    data["train"]["X"] = np.append(
        data["train"]["X"], np.atleast_2d(data["pool"]["X"][idx]), axis=0
    )

    label = int(data["pool"]["y"][idx])
    data["train"]["y"] = np.append(
        data["train"]["y"], np.atleast_1d(label), axis=0
    )

    # Track per-class and above/below-threshold counts
    selected_classes[strategy][str(label)] += 1
    if label < threshold:
        selected_classes[strategy]["below"] += 1
    else:
        selected_classes[strategy]["above"] += 1

    data["pool"]["X"] = np.delete(data["pool"]["X"], idx, axis=0)
    data["pool"]["y"] = np.delete(data["pool"]["y"], idx, axis=0)


def fit_model(base_data, paradigm, strategy, n_iterations, selected_classes):
    data = copy.deepcopy(base_data)

    scores = np.zeros(n_iterations)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")

    for i in range(n_iterations):
        model.fit(data["train"]["X"], data["train"]["y"])

        prob = model.predict_proba(data["pool"]["X"])
        scores[i] = model.score(data["test"]["X"], data["test"]["y"])

        if paradigm == "active learning":
            uncertainty = evaluate_uncertainty(prob, strategy)
            idx = int(uncertainty.argmax())
        else:
            # BUG FIX: random sampling — pick a random pool index
            idx = int(np.random.choice(len(data["pool"]["X"])))

        # BUG FIX: pass the correct strategy key so random is tracked as "random"
        update_data(data, idx, strategy=strategy, selected_classes=selected_classes)

    return scores


def build_selected_labels_df(selected_classes, dataset_name):
    """Convert the selected_classes tracking dict into a tidy DataFrame."""
    rows = []
    for strategy, counts in selected_classes.items():
        row = {"dataset": dataset_name, "strategy": strategy}
        # Per-class label counts (0–9)
        for c in range(10):
            row[f"selected_class_{c}"] = counts.get(str(c), 0)
        row["selected_below_threshold"] = counts.get("below", 0)
        row["selected_above_threshold"] = counts.get("above", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(data, name, all_dfs):
    # BUG FIX: per-experiment selected_classes dict — not global
    selected_classes = defaultdict(lambda: defaultdict(int))

    # BUG FIX: correct strategy strings assigned to each variable
    scores_al_entropy        = np.zeros((N_AVG, N_ITERATIONS))
    scores_al_least_confident = np.zeros((N_AVG, N_ITERATIONS))
    scores_al_margin         = np.zeros((N_AVG, N_ITERATIONS))
    scores_rn                = np.zeros((N_AVG, N_ITERATIONS))

    for i in tqdm(range(N_AVG), desc=f"{name} runs"):
        # BUG FIX: each variable now uses its correct strategy
        scores_al_entropy[i]         = fit_model(data, "active learning", "entropy",         N_ITERATIONS, selected_classes)
        scores_al_least_confident[i] = fit_model(data, "active learning", "least confident", N_ITERATIONS, selected_classes)
        scores_al_margin[i]          = fit_model(data, "active learning", "margin",          N_ITERATIONS, selected_classes)
        # BUG FIX: random paradigm uses strategy key "random" (not "entropy")
        scores_rn[i]                 = fit_model(data, "random",          "random",          N_ITERATIONS, selected_classes)

    # ── Plot ──────────────────────────────────────────────────────────────────
    x_axis = np.arange(N_INIT, N_INIT + N_ITERATIONS)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_axis, scores_al_entropy.mean(0),         label="AL Entropy")
    ax.plot(x_axis, scores_al_least_confident.mean(0), label="AL Least Confident")
    ax.plot(x_axis, scores_al_margin.mean(0),          label="AL Margin")
    ax.plot(x_axis, scores_rn.mean(0),                 label="Random")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"./data/{name}.png", dpi=150)
    print(f"Saved {name}.png")

    # ── DataFrames ────────────────────────────────────────────────────────────
    # 1. Dataset size & class distribution for each split
    df_info = get_dataset_info(data, dataset_name=name)
    all_dfs["dataset_info"].append(df_info)

    # 2. Which labels each strategy selected
    df_labels = build_selected_labels_df(selected_classes, dataset_name=name)
    all_dfs["selected_labels"].append(df_labels)

    print(f"\n[{name}] Dataset info:\n{df_info.to_string(index=False)}")
    print(f"\n[{name}] Selected label counts per strategy:\n{df_labels.to_string(index=False)}")


if __name__ == "__main__":
    N_INIT       = 20
    N_ITERATIONS = 80
    N_AVG        = 10

    imbalanced_data = load_data(n_init=N_INIT, imbalanced=True)
    balanced_data   = load_data(n_init=N_INIT, imbalanced=False)

    # Collect DataFrames from both experiments
    all_dfs = {"dataset_info": [], "selected_labels": []}

    run_experiment(imbalanced_data, "imbalanced", all_dfs)
    run_experiment(balanced_data,   "balanced",   all_dfs)

    # ── Combined DataFrames ───────────────────────────────────────────────────
    df_dataset_info    = pd.concat(all_dfs["dataset_info"],    ignore_index=True)
    df_selected_labels = pd.concat(all_dfs["selected_labels"], ignore_index=True)

    df_dataset_info.to_latex("dataset_info.tex", index=False, hrules=True)
    df_selected_labels.to_latex("selected_labels.tex", index=False, hrules=True)

    print("\n\n========== COMBINED DATASET INFO ==========")
    print(df_dataset_info.to_string(index=False))

    print("\n\n========== COMBINED SELECTED LABELS PER STRATEGY ==========")
    print(df_selected_labels.to_string(index=False))

    # Optionally save to CSV
    df_dataset_info.to_csv("./data/dataset_info.csv", index=False)
    df_selected_labels.to_csv("./data/selected_labels.csv", index=False)
    print("\nCSVs saved: dataset_info.csv, selected_labels.csv")