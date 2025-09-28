from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from Scaler import SimpleParallelScaler, MPICyclicEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mpi4py import MPI
import gc

NUM = ["passenger_count", "trip_distance", "extra", "trip_duration_min"]
CAT = ["RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
CYCLIC = ["pickup_hour", "pickup_dow", "pickup_month"]
TARGET = "total_amount"


def add_derived_columns(df):
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("Int8")
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek.astype("Int8")
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month.astype("Int8")
    return df


def read(filename):
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

    df = pd.read_csv(
        filename,
        header=0,
        usecols=FEATURES + [TARGET],
        dtype=DATA_TYPES,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        date_format="%m/%d/%Y %I:%M:%S %p",
    )
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = read("nytaxi2022.csv")
    print("Data loaded to memory")
    df = add_derived_columns(df)
    print("Derived columns added")

    for feature in CYCLIC:
        max_value = df[feature].max()
        angle = 2 * np.pi * df[feature] / max_value
        df[f"{feature}_sin"] = np.sin(angle).values.reshape(-1, 1)
        df[f"{feature}_cos"] = np.cos(angle).values.reshape(-1, 1)
        df.drop(feature)

    print(df.head())

    features = df.columns.difference([TARGET])
    X = df[features]
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42)

    scaler = ColumnTransformer(
        [("standard_scaler", StandardScaler(), [NUM + CAT])], remainder="passthrough"
    )
    scaler.fit_transform(Xtr)
    scaler.fit_transform(Xte)
    print(Xtr.head())
    # np.save("data/features_train.npy", Xtr_np)  # This TAKES up 3 GB of disk space
    # np.save("data/features_test.npy", Xte_np)
