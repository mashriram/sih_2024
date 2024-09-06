import os

import pandas as pd
import plotly.express as px
import streamlit as st

import src.chart_renderers as chart

DATA_PATH = "./data/5_yr_data"


def read_states_data(path: str) -> dict[str, pd.DataFrame]:
    record_count = 100000
    print(os.listdir(path))
    states_data: dict[str, pd.DataFrame] = dict()
    for file in os.listdir(path):
        df = pd.read_csv(path + "/" + file)
        record_count = record_count if len(df) > record_count else len(df)
        df = df.sample(record_count)
        states_data[file[:2].lower()] = df
        print(df.shape)
    return states_data


@st.cache_data
def get_cached_states_data():
    return read_states_data(DATA_PATH)


states_data = get_cached_states_data()
print(states_data.keys())


st.title("Analysis of Commodity Data")

state = st.selectbox("Select State", list(states_data.keys()))
commodity = st.selectbox(
    "Select Commodity", states_data[state].commodity.unique()
)  # ... (your options)
plot_type = st.selectbox(
    "Plot Type",
    ["scatter", "bar", "line", "stacked line", "stacked bar", "scatter single axis"],
)

scatter_dims = []
box_dims = []
selector = list(states_data[state].columns[1:])
year_wise_select = selector
year_wise_select.remove("year")
selector.remove("commodity")

numeric_cols = []
categorical_cols = []

for column in states_data[state].columns:
    if states_data[state][column].dtype == "O":
        categorical_cols.append(column)
    else:
        numeric_cols.append(column)

numeric_cols.remove("Unnamed: 0")
if plot_type == "scatter":
    scatter_dims = st.multiselect("Select Features for Scatter Plot", selector)
elif plot_type == "bar":
    box_dims = st.selectbox("Select Feature for Bar Chart", selector)
elif plot_type == "stacked line" or plot_type == "stacked bar":
    stacked_by_dims = st.selectbox(
        "Select Grouping Feature for  Line Chart", categorical_cols
    )
    stacked_col_dim = st.selectbox("Select Feature for Bar Chart", numeric_cols)
elif plot_type == "line":
    line_dim = st.selectbox("Select Feature for Bar Chart", numeric_cols)


req_df = states_data[state]
color = None

if "variety" in scatter_dims:
    color = "variety"
elif "district_name" in scatter_dims:
    color = "district_name"
elif "grade" in scatter_dims:
    color = "grade"
elif "month" in scatter_dims:
    color = "month"
elif "year" in scatter_dims:
    color = "year"

if plot_type == "scatter":
    if color:
        st.plotly_chart(
            px.scatter_matrix(
                req_df[req_df.commodity == commodity][scatter_dims], color=color
            )
        )
    else:
        if len(scatter_dims) >= 2:  # Check if scatter_dims has at least 2 elements
            st.plotly_chart(
                px.scatter(
                    req_df[req_df.commodity == commodity],
                    x=scatter_dims[0],
                    y=scatter_dims[1],
                )
            )
        else:
            st.warning("Please select at least 2 features for the scatter plot.")
elif plot_type == "bar":
    st.plotly_chart(
        px.bar(req_df[req_df.commodity == commodity], y=box_dims, color=color)
    )
elif plot_type == "line":
    chart.create_line_chart(req_df, line_dim)
elif plot_type == "stacked line":
    chart.create_stacked_line_chart(req_df, stacked_by_dims, stacked_col_dim)
elif plot_type == "stacked bar":
    chart.create_stacked_bar_chart(req_df, stacked_by_dims, stacked_col_dim)
elif plot_type == "scatter single axis":
    chart.create_scatter_single_axis(req_df, commodity)
else:
    st.write("Unsupported plot type!")

# Display summary statistics (optional)
# st.dataframe(req_df[req_df.commodity == commodity].describe())
