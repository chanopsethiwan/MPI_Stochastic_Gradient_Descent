import numpy as np
import sgd2 as sgd
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()


def load(features_file, target_file):
    X = np.load(features_file, mmap_mode="r")
    y = np.load(target_file, mmap_mode="r")
    n_rows_tr = X.shape[0]
    row_endpoints = np.linspace(0, n_rows_tr + 1, mpi_size + 1, dtype="i")
    start = row_endpoints[rank]
    end = row_endpoints[rank + 1]
    return X[start:end], y[start:end]


Xtr, ytr = load("data/features_train.npy", "data/targets_train.npy")
Xte, yte = load("data/features_test.npy", "data/targets_test.npy")

activation_func = "sigmoid"
weights, train_err, test_err = sgd.train(
    Xtr,
    Xte,
    ytr,
    yte,
    activation_func=activation_func,
    n_iter=100,
    hidden_dim=10000,
    lr=0.001,
)

# Plotting
if rank == 0:
    import matplotlib.pyplot as plt

    plt.plot(range(len(train_err)), train_err, label="Train RMSE")
    plt.plot(range(len(test_err)), test_err, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Training vs Testing Error")
    plt.show()

mse = sgd.compute_rmse_mpi(Xte, yte, weights=weights, activation_func=activation_func)
