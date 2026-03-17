import torch
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import ShuffleSplit
def load_data():

    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )


    indices = []
    threshold = 5
    amount = defaultdict(int)
    targets = training_data.targets

    print("sjkhfskdjfhlshf")

    for digit in range(10):
        digit_indices = (targets == digit).nonzero(as_tuple=True)[0]
        if digit < threshold:
            selected = digit_indices[:len(digit_indices) // 10]   # keep 10%
            amount["below"] += len(selected)
        else:
            selected = digit_indices[:len(digit_indices) // 100]  # keep 1%
            amount["above"] += len(selected)
        #print(f"For digit {digit} we select {len(selected)}")
        indices.append(selected)

    indices = torch.cat(indices)
    #print(len(indices), amount)

  
    subset_training_data = Subset(training_data, indices=indices)
    #print(len(subset_training_data))

  
    X = training_data.data[indices].float() / 255.0   # shape: (N, 28, 28)
    y = training_data.targets[indices]

    X_test = test_data.data.float() / 255.0
    y_test = test_data.targets

  
    seed = 0
    n_init = 20
    n = len(subset_training_data)
    pool_fraction = 0.5
    sss = ShuffleSplit(n_splits=1, train_size=pool_fraction, random_state=seed)
    train_idx, pool_idx = next(sss.split(X, y))

   
    data = dict(
        train=dict(
            X=X[train_idx],
            y=y[train_idx]
        ),
        pool=dict(
            X=X[pool_idx],
            y=y[pool_idx]
        ),
        test=dict(
            X=X_test,
            y=y_test
        )
    )
    return data