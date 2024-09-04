import cachetools.func
import pandas as pd
import plotly.express as px
import streamlit as st

from create_statewise_dfs import get_statewise_datasets


@cachetools.func.ttl_cache(maxsize=100, ttl=3600)
def get_cached_states_data():
    return get_statewise_datasets()


states_data = get_cached_states_data()


st.title("Analysis of Commodity Data")

state = st.selectbox("Select State", list(states_data.keys()))
commodity = st.selectbox(
    "Select Commodity", states_data[state].columns
)  # ... (your options)
plot_type = st.selectbox("Plot Type", ["scatter", "bar"])

scatter_dims = []
box_dims = []
if plot_type == "scatter":
    scatter_dims = st.multiselect(
        "Select Features for Scatter Plot", states_data[state].columns
    )
elif plot_type == "bar":
    box_dims = st.selectbox("Select Feature for Bar Chart", states_data[state].columns)

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
                req_df[req_df.commodity == commodity][scatter_dims], color=req_df[color]
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
        px.box(req_df[req_df.commodity == commodity], y=box_dims, color=color)
    )
else:
    st.write("Unsupported plot type!")

# Display summary statistics (optional)
st.dataframe(req_df[req_df.commodity == commodity].describe())
