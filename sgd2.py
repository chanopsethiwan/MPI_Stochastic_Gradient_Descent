import numpy as np
from mpi4py import MPI


# -------------------------
# Utility: batching
# -------------------------
def random_batch(X, y, batch_size=100):
    """Sample a random mini-batch from (X, y)."""
    if len(X) == 0:
        return []
    idx = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
    return list(zip(X[idx], y[idx]))


# -------------------------
# Initialization
# -------------------------
def initialize_weights(X, hidden_dim):
    """Initialize weights using Xavier/Glorot uniform initialization."""
    n_in = X.shape[1]

    limit_hidden = np.sqrt(6 / (n_in + hidden_dim))
    limit_output = np.sqrt(6 / (hidden_dim + 1))

    return {
        "weights_features": np.random.uniform(
            -limit_hidden, limit_hidden, (hidden_dim, n_in)
        ),
        "bias_features": np.zeros(hidden_dim),
        "weights_after_activation": np.random.uniform(
            -limit_output, limit_output, hidden_dim
        ),
        "bias_after_activation": 0.0,
    }


# -------------------------
# Activations
# -------------------------
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def diff_relu(x):
    return (x > 0).astype(float)


def diff_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def diff_tanh(x):
    return 1 - np.tanh(x) ** 2


ACT_FUNCS = {
    "relu": (relu, diff_relu),
    "sigmoid": (sigmoid, diff_sigmoid),
    "tanh": (tanh, diff_tanh),
}


# -------------------------
# Core operations
# -------------------------
def evaluate(weights, x, activation_func):
    """Forward pass for one input x."""
    act, _ = ACT_FUNCS[activation_func]
    z = weights["weights_features"] @ x + weights["bias_features"]
    h = act(z)
    return weights["weights_after_activation"] @ h + weights["bias_after_activation"]


def calculate_diff_and_gradients(weights, batch, activation_func):
    """Compute loss and gradients for one batch."""
    if not batch:  # empty batch
        return 0.0, {k: np.zeros_like(v) for k, v in weights.items()}

    act, act_diff = ACT_FUNCS[activation_func]

    grad_wf = np.zeros_like(weights["weights_features"])
    grad_bf = np.zeros_like(weights["bias_features"])
    grad_wa = np.zeros_like(weights["weights_after_activation"])
    grad_ba = 0.0
    diffs = []

    for x, y in batch:
        z = weights["weights_features"] @ x + weights["bias_features"]
        h = act(z)
        h_prime = act_diff(z)

        y_pred = (
            weights["weights_after_activation"] @ h + weights["bias_after_activation"]
        )
        error = y_pred - y
        diffs.append(error)

        grad_wa += error * h
        grad_ba += error

        delta_h = error * weights["weights_after_activation"] * h_prime
        grad_wf += np.outer(delta_h, x)
        grad_bf += delta_h

    batch_size = len(batch)
    gradients = {
        "grad_weights_features": grad_wf / batch_size,
        "grad_bias_features": grad_bf / batch_size,
        "grad_weights_after_activation": grad_wa / batch_size,
        "grad_bias_after_activation": grad_ba / batch_size,
    }
    loss = np.mean(np.square(diffs))
    return loss, gradients


def update_weights(weights, grads, lr):
    """Apply SGD update."""
    weights["weights_features"] -= lr * grads["grad_weights_features"]
    weights["bias_features"] -= lr * grads["grad_bias_features"]
    weights["weights_after_activation"] -= lr * grads["grad_weights_after_activation"]
    weights["bias_after_activation"] -= lr * grads["grad_bias_after_activation"]
    return weights


# -------------------------
# RMSE computation
# -------------------------
def compute_rmse_mpi(X_local, y_local, weights, activation_func):
    """Compute RMSE across all ranks in parallel."""
    comm = MPI.COMM_WORLD
    err_sq_local = 0.0
    n_local = len(X_local)

    for x, y in zip(X_local, y_local):
        y_pred = evaluate(weights, x, activation_func)
        err_sq_local += (y_pred - y) ** 2

    err_sq_global = comm.allreduce(err_sq_local, op=MPI.SUM)
    n_global = comm.allreduce(n_local, op=MPI.SUM)

    return np.sqrt(err_sq_global / max(1, n_global))


# -------------------------
# Training
# -------------------------
def train(
    Xtr_local,
    Xte_local,
    ytr_local,
    yte_local,
    activation_func="relu",
    batch_size=10,
    lr=0.001,
    hidden_dim=1000,
    n_iter=100,
    sample_test_size=100000,
):
    """
    Train one-hidden-layer NN using SGD + MPI.
    Assumes dataset partitioning is already done at a higher level.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize weights on rank 0, broadcast to all
    weights = initialize_weights(Xtr_local, hidden_dim) if rank == 0 else None
    weights = comm.bcast(weights, root=0)

    train_rmses, test_rmses = [], []
    small_Xte = Xte_local[:sample_test_size]
    small_yte = yte_local[:sample_test_size]

    for i in range(n_iter):
        # Draw batch from *local* training set
        batch = random_batch(Xtr_local, ytr_local, batch_size=batch_size)

        local_mse, local_grads = calculate_diff_and_gradients(
            weights, batch, activation_func
        )

        # Aggregate gradients and loss
        global_grads = {
            k: comm.allreduce(v, op=MPI.SUM) / size for k, v in local_grads.items()
        }
        global_mse = comm.allreduce(local_mse, op=MPI.SUM) / size
        train_rmse = np.sqrt(global_mse)

        if train_rmse < 1e-4:
            break

        weights = update_weights(weights, global_grads, lr)

        # Compute full RMSEs in parallel
        test_rmse = compute_rmse_mpi(small_Xte, small_yte, weights, activation_func)
        # train_rmse = compute_rmse_mpi(Xtr_local, ytr_local, weights, activation_func)

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

        if rank == 0:
            print(
                f"Iter {i+1}/{n_iter} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
            )

    return weights, train_rmses, test_rmses
