import numpy as np
import sgd2 as sgd
from mpi4py import MPI
import pickle
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()


def load_data(filename):
    data_arr = np.memmap("final_normalized.npy", dtype="float32", mode="r")
    data_arr = data_arr.reshape((-1, 13))
    row_endpoints = np.linspace(0, data_arr.shape[0] + 1, mpi_size + 1, dtype="i")
    start = row_endpoints[rank]
    end = row_endpoints[rank + 1]
    mid = int(start + 0.7 * (end - start))
    # Return Xtr, ytr, Xte, yte
    return (
        data_arr[start:mid, :-1],
        data_arr[start:mid, -1],
        data_arr[mid:end, :-1],
        data_arr[mid:end, -1],
    )


def load(features_file, target_file):
    X = np.load(features_file, mmap_mode="r")
    y = np.load(target_file, mmap_mode="r")
    n_rows_tr = X.shape[0]
    row_endpoints = np.linspace(0, n_rows_tr + 1, mpi_size + 1, dtype="i")
    start = row_endpoints[rank]
    end = row_endpoints[rank + 1]
    return X[start:end], y[start:end]


Xtr, ytr, Xte, yte = load_data("final_normalized.npy")

activation_funcs = ["sigmoid", "tanh", "relu"]
batch_sizes = [1, 5, 10, 50, 100]

results = []

for activation_func in activation_funcs:
    for batch_size in batch_sizes:
        start_time = time.time()
        if rank == 0:
            print(f"Training with {activation_func} and batch_size={batch_size}")
        weights, train_err, test_err = sgd.train(
            Xtr,
            Xte,
            ytr,
            yte,
            activation_func=activation_func,
            max_n_iter=100,
            min_n_iter=5,
            hidden_dim=10_000,
            lr=0.01,
            batch_size=batch_size,
        )
        time_elapsed = time.time() - start_time
        if rank == 0:
            print(f"That took {time_elapsed:.2f} seconds")
        # rmse = sgd.compute_rmse_mpi(
        #     Xte, yte, weights=weights, activation_func=activation_func
        # )
        rmse = None
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
            with open("results3.pkl", "wb") as f:
                pickle.dump(results, f)
