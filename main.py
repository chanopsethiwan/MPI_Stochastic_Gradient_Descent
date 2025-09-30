import numpy as np
import sgd2 as sgd
from mpi4py import MPI
import pickle
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()


def load_data(filename: str):
    data_arr = np.memmap("final_normalized.npy", dtype="float32", mode="r")
    data_arr = data_arr.reshape((-1, 15))
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


Xtr, ytr, Xte, yte = load_data("final_normalized.npy")

activation_funcs = ["relu", "tanh", "sigmoid"]
batch_sizes = [1, 5, 10, 50, 100]

results = []

for activation_func in activation_funcs:
    for batch_size in batch_sizes:
        hidden_dim = 1000
        # # For fairness, everyone starts with the same initial weights.
        # initial_weights = sgd.initialize_weights(Xtr, hidden_dim) if rank == 0 else None
        # initial_weights = comm.bcast(initial_weights, root=0)
        if rank == 0:
            print(f"Training with {activation_func} and batch_size={batch_size}")
        start_time = time.time()
        weights, loss, test_err = sgd.train(
            Xtr,
            ytr,
            activation_func=activation_func,
            max_n_iter=100,
            min_n_iter=10,
            hidden_dim=10_000,
            lr=0.01,
            batch_size=batch_size,
            sample_test_size=10_000,
        )
        time_elapsed = time.time() - start_time
        if rank == 0:
            print(f"That took {time_elapsed:.2f} seconds")

        # Compute train_rmse
        train_rmse_start_time = time.time()
        train_rmse = sgd.compute_rmse_mpi(
            Xtr,
            ytr,
            weights=weights,
            activation_func=activation_func,
            show_progress=True,
            chunk_size=2**11,
        )
        train_rmse_duration = time.time() - train_rmse_start_time
        print(f"The train_rmse is {train_rmse:2f}")

        # Compute test_rmse
        test_rmse_start_time = time.time()
        test_rmse = sgd.compute_rmse_mpi(
            Xte,
            yte,
            weights=weights,
            activation_func=activation_func,
            show_progress=True,
            chunk_size=2**11,
        )
        test_rmse_duration = time.time() - test_rmse_start_time
        print(f"The test_rmse is {test_rmse:2f}")

        results.append(
            {
                "activation_func": activation_func,
                "batch_size": batch_size,
                "loss": loss,
                "test_err": test_err,
                "test_rmse": test_rmse,
                "time_elapsed": time_elapsed,
                "test_rmse_duration": test_rmse_duration,
                "train_rmse": train_rmse,
                "train_rmse_duration": train_rmse_duration,
            }
        )
        if rank == 0:
            with open("results5.pkl", "wb") as f:
                pickle.dump(results, f)
            with open(f"weights/{activation_func}-{batch_size}.pkl", "wb") as f:
                pickle.dump(weights, f)
