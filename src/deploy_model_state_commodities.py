# %%
DATA_SOURCE = "/home/ubuntu/sih_2024_project/sih_2024_data_source/statewise_results/"

# %%
import pandas as pd

# %%
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import plotly.express as px

# %%
scores = dict()


def get_scores(y_test, y_pred):
    r2_ = r2_score(y_test, y_pred)
    rmse_ = root_mean_squared_error(y_test, y_pred)
    mae_ = mean_absolute_error(y_test, y_pred)
    return {"r2": r2_, "mae": mae_, "rmse": rmse_}


# %%
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, DeepAR, TFT, LSTM, RNN, GRU
from neuralforecast.losses.pytorch import DistributionLoss, MAE, MSE, MAPE, SMAPE
import torch

from darts import TimeSeries
from darts.models import (
    NHiTSModel,
)


def create_darts_models(input_chunk_length=120, output_chunk_length=30, n_epochs=100):
    """
    Create a collection of Darts models with correct parameters
    """
    # Common parameters for neural networks
    nn_params = {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "n_epochs": n_epochs,
        "batch_size": 32,
        "force_reset": True,
    }

    models = {
        "nhits": NHiTSModel(
            **nn_params,
            num_stacks=3,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            pooling_kernel_sizes=None,
            n_freq_downsample=None,
            dropout=0.1,
            activation="ReLU",
            MaxPool1d=True,
        ),
    }

    return models


def train_and_forecast(df_train, df_test):
    """
    Train models and generate forecasts using either Nixtla or Darts
    """

    # Darts workflow
    # Convert pandas DataFrame to Darts TimeSeries
    df = pd.concat([df_train, df_test], axis=1)
    series = TimeSeries.from_dataframe(
        df, "ds", "y", fill_missing_dates=True, freq=None
    )

    # Create and train models
    models = create_darts_models()
    forecasts = {}

    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(series)
        forecast = model.predict(len(df_test))

        forecasts[name] = {"data": forecast, "model": model}

    return forecasts


# %%


def create_interpolated_ranges(dataframe, date_col, value_col):
    dataframe[date_col] = pd.to_datetime(dataframe[date_col])
    date_range = pd.date_range(
        start=dataframe[date_col].min(), end=dataframe[date_col].max()
    )
    full_df = pd.DataFrame({date_col: date_range})
    merged_df = pd.merge(full_df, dataframe, on=date_col, how="left")
    merged_df[value_col] = merged_df[value_col].interpolate()
    merged_df[value_col] = (
        merged_df[value_col].fillna(method="bfill").fillna(method="ffill")
    )
    return merged_df


# %%
from ast import mod
import os
from re import I

ers = {}
for commodity in os.listdir(DATA_SOURCE):
    print(commodity)
    ers[commodity] = {}
    path = DATA_SOURCE + commodity
    for state_csv in os.listdir(path):
        sub_path = path + "/" + state_csv
        state = state_csv.partition("_")[0]
        df = pd.read_csv(sub_path)
        # df['datetime'] = pd.to_datetime(df['date'])
        df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
        df.sort_values(by="datetime", ascending=True, inplace=True)
        # print(df.head())
        TRAIN_LEN = int(0.8 * len(df))
        df_train, df_test = (df[:TRAIN_LEN], df[TRAIN_LEN:])
        if df_train.shape[0] < 151 or df_test.shape[0] < 15:
            continue
        df_train.set_index("datetime", inplace=True)
        df_train.sort_index(inplace=True)
        df_test.set_index("datetime", inplace=True)
        df_test.sort_index(inplace=True)
        df_train_dt = df_train.groupby("datetime").agg({"modal_rs_quintal": "mean"})
        df_test_dt = df_test.groupby("datetime").agg({"modal_rs_quintal": "mean"})
        df_train_dt.reset_index(inplace=True)
        df_train_dt.rename(
            columns={"datetime": "ds", "modal_rs_quintal": "y"}, inplace=True
        )
        df_test_dt.reset_index(inplace=True)
        df_test_dt.rename(
            columns={"datetime": "ds", "modal_rs_quintal": "y"}, inplace=True
        )
        ######
        print(df_test_dt.shape, df_train_dt.shape)
        print(df_train_dt["ds"].unique())
        df_train_dt = create_interpolated_ranges(df_train_dt, "ds", "y")
        df_test_dt = create_interpolated_ranges(df_test_dt, "ds", "y")
        print(df_train_dt.head(20))
        if df_train_dt.shape[0] < 151 or df_test_dt.shape[0] < 15:
            continue
        nhits_forecast = train_and_forecast(df_train=df_train_dt, df_test=df_test_dt)
        print(nhits_forecast)
        for name, data_model in nhits_forecast.items():
            os.makedirs(f"./model_results/{commodity}/{state}/", exist_ok=True)
            nhits_forecast[name]["model"].save(
                f"./model_results/{commodity}/{state}/nhits.pkt"
            )
            nhits_forecast[name] = pd.DataFrame(nhits_forecast[name]["data"].values())[
                0
            ]
        result = pd.DataFrame(nhits_forecast)
        result_y = df_test_dt["y"]
        results = pd.concat([result, result_y], axis=1)
        for column in results.columns:
            scores[column] = get_scores(results["y"], results[column])

        results.to_csv(f"./model_results/{commodity}/results.csv")
        error_results = pd.DataFrame(scores)
        error_results.to_csv(f"./model_results/{commodity}/errors.csv")
        # ers[state] = {"results": results, "error_results": error_results}
        # px.line(
        #     results,
        #     x=results.index,
        #     y=[
        #         "y",
        #         "nhits",
        #     ],
        # )


# %%


# %%


# %%
