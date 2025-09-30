# MPI SGD

## 0. Prerequisites

- Install **_openiMPI_**:

```bash
brew install open-mpi
```

or

```bash
sudo apt install open-mpi
```

- Then, install **_mpi4py_** inside your Python environment:

```bash
pip install mpi4py
```

## 1. Data Preprocessing

- Download the `nytaxi2022.csv` file and put it in the project folder (next to `main.py`).
- Run `preprocessing.ipynb` to get `final_normalized.npy`

## 2. Run

```bash
mpiexec -n 3 python main.py
```
