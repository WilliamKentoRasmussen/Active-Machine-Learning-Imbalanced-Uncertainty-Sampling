import torch
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import ShuffleSplit

def load_data_v2(n_init= 100):

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

    

    for digit in range(10):
        digit_indices = (targets == digit).nonzero(as_tuple=True)[0]
        if digit < threshold:
            selected = digit_indices[:len(digit_indices) // 800]   # keep 10%
            amount["below"] += len(selected)
        else:
            selected = digit_indices[:len(digit_indices) // 2000]  # keep 1%
            amount["above"] += len(selected)
        print(f"For digit {digit} we select {len(selected)}")
        indices.append(selected)

    indices = torch.cat(indices)
    #print(len(indices), amount)

  
    subset_training_data = Subset(training_data, indices=indices)
    #print(len(subset_training_data))

  
    X = training_data.data[indices].float() / 255.0 
    y = training_data.targets[indices]

    
    X_test = test_data.data.float() / 255.0
    y_test = test_data.targets
    X_test= X_test[0:100]
    y_test = y_test[0:100]

  
    seed = 0
    n = len(subset_training_data)
    

    sss = ShuffleSplit(n_splits=1, train_size=n_init/n, random_state=seed)
    train_idx, pool_idx = next(sss.split(X, y))

   
    data = dict(
        train=dict(
            X=X[train_idx].reshape(len(X[train_idx]), -1),
            y=y[train_idx]
        ),
        pool=dict(
            X=X[pool_idx].reshape(len(X[pool_idx]), -1),
            y=y[pool_idx]
        ),
        test=dict(
            X=X_test.reshape(len(X_test), -1),
            y=y_test
        )
    )
    return data

data = load_data_v2(10) #n_init
print(f"Train X len: {len(data["train"]["X"])}")
print(f"Train y len: {len(data["train"]["y"])}\n")

print(f"Pool X len: {len(data["pool"]["X"])}")
print(f"Pool y len: {len(data["pool"]["y"])}\n")

print(f"Test X len: {len(data["test"]["X"])}")
print(f"Test y len: {len(data["test"]["y"])}\n")