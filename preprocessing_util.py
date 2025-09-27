from Scaler import SimpleParallelScaler, MPICyclicEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUM = ["passenger_count", "trip_distance", "extra", "trip_duration_min"]
CAT = ["RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
CYCLIC = ["pickup_hour", "pickup_dow", "pickup_month"]
TARGET = "total_amount"


def preprocessing_pipeline(comm, X):
    """
    Normal scaling for numeric features and cyclic encoding for cyclic features.
    """
    # preprocess NUM features using StandardScaler for NUM + CAT for now, One hot encoding cause memory
    # overload issue
    scaler = SimpleParallelScaler(comm=comm)
    X_num = scaler.fit_transform(X[NUM + CAT])
    # cyclic encoding for cyclic features
    cyclic_encoder = MPICyclicEncoder(comm=comm, drop_original=True)
    X_cyclic = cyclic_encoder.fit_transform(X[CYCLIC])
    return np.hstack([X_num, X_cyclic], dtype="float64")


def split(df):
    X, y = df[NUM + CAT + CYCLIC], df[TARGET].to_numpy()
    return train_test_split(X, y, test_size=0.30, random_state=42)


def read_csv(filename, rows_to_read):
    DATA_TYPES = {
        "tpep_pickup_datetime": "int",
        "tpep_dropoff_datetime": "int",
        "passenger_count": "Int8",
        "trip_distance": "float32",
        "RatecodeID": "category",
        "PULocationID": "category",
        "DOLocationID": "category",
        "payment_type": "category",
        "extra": "float32",
        "total_amount": "float32",
    }
    FEATURES = list(DATA_TYPES.keys())

    row_start, row_end = rows_to_read
    df = pd.read_csv(
        filename,
        header=0,
        skiprows=np.arange(1, row_start, dtype="i"),
        nrows=row_end - row_start,
        usecols=FEATURES + [TARGET],
        dtype=DATA_TYPES,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        date_format="%m/%d/%Y %I:%M:%S %p",
    )
    df.dropna(inplace=True)
    return df


def add_derived_columns(df):
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("Int8")
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek.astype("Int8")
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month.astype("Int8")
    return df
