import plotly.express as px
from dash import Dash, Input, Output, dcc, dependencies, html

from create_statewise_dfs import get_statewise_datasets

states_data: dict | None = None
app = Dash(__name__)


states_data = get_statewise_datasets() if states_data is None else states_data

app.layout = html.Div(
    [
        html.H4("Analysis of Commodity  Data"),
        dcc.Dropdown(
            id="state",
            options=list(states_data.keys()),
            value="ap",
        ),
        dcc.Dropdown(
            id="commodity",
            options=[
                "Gram dal",
                "gur",
                "moong dal",
                "onion",
                "Rice",
                "tea",
                "tur dal",
                "vanaspati",
                "Groundnut oil",
                "masur dal",
                "mustard oil",
                "Potato",
                "Sugar",
                "tomato",
                "urad dal",
                "wheat",
            ],
            value="Rice",
        ),
        dcc.Dropdown(
            id="plot_type",
            options=["scatter", "bar"],
            value="scatter",
        ),
        html.Div(
            id="scatter-dropdown-container",
            children=[
                dcc.Dropdown(
                    id="scatter-multi-dropdown",
                    options=states_data["ap"].columns,
                    value=[
                        "modal_rs_quintal",
                        "variety",
                    ],  # Exclude year initially (optional)
                    multi=True,
                )
            ],
            style={"display": "none"},  # Initially hide the dropdown
        ),
        html.Div(
            id="box-dropdown-container",
            children=[
                dcc.Dropdown(
                    id="box-dropdown",
                    options=states_data["ap"].columns,
                    value=[
                        "modal_rs_quintal",
                    ],  # Exclude year initially (optional)
                )
            ],
            style={"display": "none"},  # Initially hide the dropdown
        ),
        dcc.Graph(id="graph"),
    ]
)


@app.callback(
    dependencies.Output("scatter-dropdown-container", "style"),
    [dependencies.Input("plot_type", "value")],
)
def update_scatter_dropdown_visibility(plot_type):
    if plot_type == "scatter":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    dependencies.Output("box-dropdown-container", "style"),
    [dependencies.Input("plot_type", "value")],
)
def update_box_dropdown_visibility(plot_type):
    if plot_type == "bar":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("graph", "figure"),
    [
        Input("state", "value"),
        Input("commodity", "value"),
        Input("plot_type", "value"),
        Input("scatter-multi-dropdown", "value"),
        Input("box-dropdown", "value"),
    ],
)
def update_chart(state, commodity, plot_type, scatter_dims, box_dims):
    req_df = states_data[state]
    color: str | None = None
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
        fig = px.scatter_matrix(
            req_df[req_df.commodity == commodity][scatter_dims], color=req_df[color]
        )  # Use scatter matrix for scatter plots
    elif plot_type == "bar":
        #
        fig = px.box(
            data_frame=req_df[req_df.commodity == commodity],
            y=states_data[state][box_dims],
            color=color,
        )
        # Cleanup temporary column (optional)
    else:
        # Handle other plot types (optional)
        fig = None
    return fig


if __name__ == "__main__":
    app.run(debug=True)
