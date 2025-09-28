import numpy as np
import sgd2 as sgd
from mpi4py import MPI
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import time

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
print(Xtr.shape)

activation_funcs = ["sigmoid", "tanh", "relu"]
batch_sizes = [1, 5, 10, 50, 100]

activation_func = "sigmoid"
results = []

for activation_func in activation_funcs:
    for batch_size in batch_sizes:
        start_time = None
        if rank == 0:
            start_time = time.time()
            print(f"Training with {activation_func} and batch_size={batch_size}")
        weights, train_err, test_err = sgd.train(
            Xtr,
            Xte,
            ytr,
            yte,
            activation_func=activation_func,
            n_iter=100,
            hidden_dim=1000,
            lr=0.001,
            batch_size=batch_size,
        )
        if rank == 0:
            rmse = sgd.compute_rmse_mpi(
                Xte, yte, weights=weights, activation_func=activation_func
            )
            results.append(
                {
                    "activation_func": activation_func,
                    "batch_size": batch_size,
                    "train_err": train_err,
                    "test_err": test_err,
                    "mse": rmse,
                    "time_elapsed": time.time() - start_time,
                }
            )

if rank == 0:
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
