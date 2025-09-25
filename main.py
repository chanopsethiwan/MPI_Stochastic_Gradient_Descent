from mpi4py import MPI
import sgd_functions as sgd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import (
    preprocessing_pipeline,
    read_csv,
    add_derived_columns,
    split,
)

FILENAME = "nytaxi2022.csv"
# FILENAME = "nytaxi_mini.csv"  # For testing
activation_func = "relu"
batch_size = 1000
LEARNING_RATE = 0.001

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

if rank == 0:
    # Count the number of Lines
    with open(FILENAME, "rb") as f:
        num_lines = sum(1 for _ in f)
    row_endpoints = np.linspace(0, num_lines, mpi_size + 1, dtype="i")
else:
    row_endpoints = None
row_endpoints = comm.bcast(row_endpoints, root=0)

print(f"{rank} received: {row_endpoints}")
df = read_csv(FILENAME, (row_endpoints[rank], row_endpoints[rank + 1]))
add_derived_columns(df)

# Train-test Split
Xtr, Xte, ytr, yte = split(df)
Xtr = preprocessing_pipeline(comm, Xtr)
Xte = preprocessing_pipeline(comm, Xte)

# Neural Network time
# initialization
weights = None
test_rmses = []
train_rmses = []
if rank == 0:
    weights = sgd.initialise_weights(Xtr, 1000)
weights = comm.bcast(weights, root=0)

# Steps
for _ in range(100):
    batch = sgd.random_batch(Xtr, ytr, batch_size=batch_size)

    local_mse, local_gradients = sgd.calculate_diff_and_gradients(
        weights, batch, activation_func=activation_func
    )

    global_gradients = {}
    for key in local_gradients:
        global_gradients[key] = (
            comm.allreduce(local_gradients[key], op=MPI.SUM) / mpi_size
        )
    global_mse = comm.allreduce(local_mse, op=MPI.SUM) / mpi_size
    rmse = np.sqrt(global_mse)

    if rmse < 1e-4:
        break
    weights = sgd.update_weights(weights, global_gradients, learning_rate=LEARNING_RATE)
    test_batch = sgd.random_batch(Xte, yte)
    test_loss = sgd.calculate_loss(weights, test_batch, activation_func)
    global_test_loss = comm.allreduce(test_loss, op=MPI.SUM) / mpi_size
    if rank == 1:
        print(f"{rmse}\t {np.sqrt(global_test_loss)}")
        train_rmses.append(rmse)
        test_rmses.append(np.sqrt(global_test_loss))

if rank == 1:
    plt.title("Testing RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(range(len(test_rmses)), test_rmses, label="Testing RMSE")
    plt.show()
    plt.title("Training RMSE")
    plt.plot(range(len(train_rmses)), train_rmses, label="Training RMSE")
    plt.show()
