import datetime
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from darts.models import NBEATSModel, NHiTSModel

DATA_SOURCE = "../../../sih_2024_data_source/statewise_results/"
MODEL_SOURCE = "../model_results/"


def load_data(commodity, state):
    # Code to load the data for the selected commodity and state
    try:
        data_file = os.path.join(DATA_SOURCE, f"{commodity}/{state}_all_years.csv")
        data = pd.read_csv(data_file)
        try:
            data.drop(columns={"Unnamed :0"}, inplace=True)
        except Exception:
            print("clean")
        data = data.groupby("datetime").agg({"modal_rs_quintal": "mean"}).reset_index()
        data.columns = ["datetime", "modal_rs_quintal"]
        data.set_index("datetime", inplace=True)
        data.sort_index(ascending=True, inplace=True)

        return data
    except Exception as e:
        print(e)


def load_model(commodity, state):
    # Code to load the pre-trained NHITS model for the selected commodity and state
    try:
        model_file = os.path.join(MODEL_SOURCE, f"{commodity}/{state}/nhits.pkt")
        model = NHiTSModel.load(model_file)
        return model
    except Exception:
        st.write(f"{state} has no data for selected commodity")


def generate_prediction(model):
    # Code to use the loaded model to generate 100-day predictions
    if model != None:
        prediction = model.predict(n=100)
        return prediction
    else:
        print("other commodity")


def plot_results(data, prediction):
    # Code to create the Plotly line chart with historical data and prediction
    if data.empty or prediction == None:
        print("None")
    else:
        prediction_df = prediction.pd_dataframe().reset_index()
        date_min = prediction_df["ds"].min()
        data = data[pd.to_datetime(data.index) < date_min]
        # data = data[data.shape[0] - 180 :]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["modal_rs_quintal"],
                mode="lines",
                name="Historical Data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prediction_df.ds,
                y=prediction_df.y,
                mode="lines",
                name="Prediction",
            )
        )
        fig.update_layout(
            title=f"{selected_commodity} - {selected_state}",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(fig)


def get_errors(commodity):
    try:
        err = pd.read_csv(MODEL_SOURCE + commodity + "/errors.csv")
        err.index = ["r2", "mae", "rmse"]
        st.write(f"The error for  {commodity} for all states combined")
        st.write(err.nhits)
    except Exception:
        st.write(f"No errors for {commodity}")


st.title("Commodity Price Prediction")
st.write(
    "This app provides 100-day price predictions for different commodities and states."
)

# Get the list of available commodities and states
commodities = os.listdir(DATA_SOURCE)
commodities_mapper: dict[str, str] = {
    commodity.capitalize(): commodity for commodity in commodities
}
# Sidebar for user input
selected_commodity = st.sidebar.selectbox(
    "Select Commodity", list(commodities_mapper.keys())
)
selected_commodity = commodities_mapper[selected_commodity]
states = [
    os.path.splitext(file)[0].split("_")[0]
    for file in os.listdir(DATA_SOURCE + selected_commodity)
]
state_mapper: dict[str, str] = {
    "Arunachal Pradesh": "AR",
    "Andaman & Nicobar": "AN",
    "Andhra Pradesh": "AP",
    "Assam": "AS",
    "Bihar": "BI",
    "Chhattisgarh": "CG",
    "Delhi": "DL",
    "Chandigarh": "CH",
    "Goa": "GO",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jharkhand": "JR",
    "Jammu  &  Kashmir and Ladakh": "JK",
    "Karnataka": "KK",
    "Kerala": "KL",
    "Madhya Pradesh": "MP",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "MG",
    "Mizoram": "MZ",
    "Nagaland": "NG",
    "Odisha": "OR",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TL",
    "Tripura": "TR",
    "Uttarakhand": "UC",
    "Uttar Pradesh": "UP",
    "West Bengal": "WB",
}
state_ac_mapper = {
    "AR": "Arunachal Pradesh",
    "AN": "Andaman & Nicobar",
    "AP": "Andhra Pradesh",
    "AS": "Assam",
    "BI": "Bihar",
    "CG": "Chhattisgarh",
    "DL": "Delhi",
    "CH": "Chandigarh",
    "GO": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JR": "Jharkhand",
    "JK": "Jammu  &  Kashmir and Ladakh",
    "KK": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "MG": "Meghalaya",
    "MZ": "Mizoram",
    "NG": "Nagaland",
    "OR": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TL": "Telangana",
    "TR": "Tripura",
    "UC": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}
selected_state = st.sidebar.selectbox(
    "Select State", [state_ac_mapper[state] for state in states]
)
selected_state = state_mapper[selected_state]

# Load data and generate prediction
data = load_data(selected_commodity, selected_state)
model = load_model(selected_commodity, selected_state)
prediction = generate_prediction(model)

# Display results
plot_results(data, prediction)
get_errors(selected_commodity)
