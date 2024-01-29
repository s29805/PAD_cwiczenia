from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
# import dash_design_kit as ddk
import plotly.graph_objects as go
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np

app = Dash(__name__)

dt = pd.read_csv("messy_data.csv")
dt = dt.rename(columns={'carat': 'carat', 
                        ' clarity': 'clarity', 
                        ' color': 'color', 
                        ' cut': 'cut', 
                        ' x dimension': 'x_dimension', 
                        ' y dimension': 'y_dimension', 
                        ' z dimension': 'z_dimension', 
                        ' depth': 'depth', 
                        ' table': 'table', 
                        ' price': 'price'
                        })
dt.replace(" ", np.NaN, inplace=True)
dt.drop_duplicates(list(dt.columns), inplace=True)
# dt.head(25)
# x, y, z i carat wysokie korelacje, bazują na sobie, table i depth zamieniam na średnią ponieważ jest mała różnica między wierszami, przy braku price usuwam, ponieważ są to tylko 4 wartości
dt[['x_dimension', 'y_dimension', 'z_dimension', 'table', 'price', 'depth']] = dt[['x_dimension', 'y_dimension', 'z_dimension', 'table', 'price', 'depth']].apply(pd.to_numeric)
dt['x_dimension'] = dt.apply(lambda row: row['y_dimension'] if np.isnan(row['x_dimension']) else row['x_dimension'], axis=1)
dt['y_dimension'] = dt.apply(lambda row: row['x_dimension'] if np.isnan(row['y_dimension']) else row['y_dimension'], axis=1)
z_to_x_and_y_correlation = dt['z_dimension'] / ((dt['x_dimension'] + dt['y_dimension']) / 2)
dt['z_dimension'] = dt.apply(lambda row: round((row['x_dimension'] + row['y_dimension']) / 2 * np.mean(z_to_x_and_y_correlation), 2) if np.isnan(row['z_dimension']) else row['z_dimension'], axis=1)
carat_to_x_correlation = dt['carat'] / dt['x_dimension']
dt['carat'] = dt.apply(lambda row: round(row['x_dimension'] * np.mean(carat_to_x_correlation), 2) if np.isnan(row['carat']) else row['carat'], axis=1)
dt['depth'].replace(np.NaN, round(np.mean(dt['depth']), 2), inplace=True)
dt['table'].replace(np.NaN, round(np.mean(dt['table']), 2), inplace=True)
dt['color'] = dt['color'].apply(lambda x: x.upper())
dt['clarity'] = dt['clarity'].apply(lambda x: x.upper())
dt['cut'] = dt['cut'].apply(lambda x: x.upper())
dt.dropna(subset=['price'], inplace=True)

# zamiana znaczników na dane numeryczne
le_clarity = LabelEncoder()
le_color = LabelEncoder()
le_cut = LabelEncoder()
dt['clarity'] = le_clarity.fit_transform(dt['clarity'])
dt['color'] = le_color.fit_transform(dt['color'])
dt['cut'] = le_cut.fit_transform(dt['cut'])

dt_with_outliers = dt


#depth
Q1_depth = np.percentile(dt['depth'], 25, method='linear')
Q3_depth = np.percentile(dt['depth'], 75, method='linear')
IQR_depth = Q3_depth - Q1_depth
upper_treshold_depth = Q3_depth + 1.5 * IQR_depth
lower_treshold_depth = Q1_depth - 1.5 * IQR_depth

#carat
Q1_carat = np.percentile(dt['carat'], 25, method='linear')
Q3_carat = np.percentile(dt['carat'], 75, method='linear')
IQR_carat = Q3_carat - Q1_carat
upper_treshold_carat = Q3_carat + 1.5 * IQR_carat
lower_treshold_carat = Q1_carat - 1.5 * IQR_carat

#price
Q1_price = np.percentile(dt['price'], 25, method='linear')
Q3_price = np.percentile(dt['price'], 75, method='linear')
IQR_price = Q3_price - Q1_price
upper_treshold_price = Q3_price + 1.5 * IQR_price
lower_treshold_price = Q1_price - 1.5 * IQR_price

# Ucinanie wartości odstających dla kategorii depth, czrat i price do zakresu wąsów wykresu pudełkowego
dt['depth'] = dt['depth'].apply(lambda row: Q3_depth if row > Q3_depth else Q1_depth if row < Q1_depth else row)
dt['carat'] = dt['carat'].apply(lambda row: Q3_carat if row > Q3_carat else Q1_carat if row < Q1_carat else row)
dt['price'] = dt['price'].apply(lambda row: Q3_price if row > Q3_price else Q1_price if row < Q1_price else row)



app.layout = html.Div([
    html.H1(),
    html.Hr(),
    html.H1("Sebastian Szczepański S29805 PAD Projekt - interaktywny dashboard"),
    html.Hr(),
    html.H1(),
    html.H2("Tabela z próbką danych"),
    html.H1(),
    html.Div([
        dash_table.DataTable(dt.head(15).to_dict('records'), [{"name": i, "id": i} for i in dt.columns])
    ]),
    html.H1(),
    html.Hr(),
    html.H1(),
    html.H2("Wykresy pudełkowe atrybutów"),
    html.H1(),
    dcc.Graph(id='indicator-boxplots'),
    html.H1(),
    html.Hr(),
    html.H1(),
    html.H2("Histogramy przedstawiające liczebności kategorii"),
    html.H1(),
    dcc.Graph(id='indicator-hist'),
    html.H1(),
    html.Hr(),
    html.H1(),
    html.H2("Wykres wybranego atrybutu od atrybutu decyzyjnego"),
    html.H1(),
    html.Div([
        html.Div([
            dcc.Dropdown(
                dt.columns[:-1],
                'carat',
                id='xaxis-column'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='xaxis-type',
                inline=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
    dcc.Graph(id='indicator-graphic')
    ]),
    html.H1(),
    html.Hr(),
    html.H1(),
    html.H2("Model regresji wybranego atrybutu"),
    html.H1(),
    html.Div([
        html.Div([
            dcc.Dropdown(
                dt.columns[:-1],
                'carat',
                id='xaxis-column-reg'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='xaxis-type-reg',
                inline=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
    dcc.Graph(id='indicator-reg'),
    ]),
    html.H1(),
    html.Hr(),
    html.H1()
])


@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    )
def update_graph(xaxis_column_name, xaxis_type):
    fig = px.scatter(x=dt[xaxis_column_name], y=dt["price"])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title="price",
                     type='linear')

    return fig

@app.callback(
    Output('indicator-reg', 'figure'),
    Input('xaxis-column-reg', 'value'),
    Input('xaxis-type-reg', 'value'),
    )
def update_reg_graph(xaxis_column_name, xaxis_type):
    fig = px.scatter(x=dt[xaxis_column_name], y=dt["price"])
    formula = f"price ~ {xaxis_column_name}"
    lm = smf.ols(formula, data=dt)
    lm_fit = lm.fit()
    (b0, b1) = lm_fit.params
    y = list(dt[xaxis_column_name].apply(lambda x: float(x) * b1 + b0))
    fig.add_trace(go.Line(x=list(dt[xaxis_column_name]), y=y, name="Line model"))
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title="price",
                     type='linear')

    return fig

@app.callback(
    Output('indicator-boxplots', 'figure'),
    Input('xaxis-column', 'value'))
def boxplot_configuration(xaxis_column):

    fig = make_subplots(rows=2, cols=5)
    fig.add_trace(go.Box(y=list(dt_with_outliers["carat"]), name="carat"), row=1, col=1)
    fig.add_trace(go.Box(y=list(dt_with_outliers["x_dimension"]), name="x_dimension"), row=1, col=2)
    fig.add_trace(go.Box(y=list(dt_with_outliers["y_dimension"]), name="y_dimension"), row=1, col=3)
    fig.add_trace(go.Box(y=list(dt_with_outliers["z_dimension"]), name="z_dimension"), row=1, col=4)
    fig.add_trace(go.Box(y=list(dt_with_outliers["cut"]), name="cut"), row=1, col=5)
    fig.add_trace(go.Box(y=list(dt_with_outliers["clarity"]), name="clarity"), row=2, col=1)
    fig.add_trace(go.Box(y=list(dt_with_outliers["color"]), name="color"), row=2, col=2)
    fig.add_trace(go.Box(y=list(dt_with_outliers["table"]), name="table"), row=2, col=3)
    fig.add_trace(go.Box(y=list(dt_with_outliers["depth"]), name="depth"), row=2, col=4)
    fig.add_trace(go.Box(y=list(dt_with_outliers["price"]), name="price"), row=2, col=5)
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout(margin={'l': 80, 'b': 80, 't': 10, 'r': 0})
    # edit axis labels
    # fig['layout']['xaxis']['title']='Label x-axis 1'
    # fig['layout']['xaxis2']['title']='Label x-axis 2'
    # fig['layout']['yaxis']['title']='Label y-axis 1'
    # fig['layout']['yaxis2']['title']='Label y-axis 2'
    return fig

@app.callback(
    Output('indicator-hist', 'figure'),
    Input('xaxis-column', 'value'))
def boxplot_configuration(xaxis_column):
    fig = make_subplots(rows=2, cols=5)
    fig.add_trace(go.Histogram(x=list(dt["carat"]), name="carat"), row=1, col=1)
    fig.add_trace(go.Histogram(x=list(dt["x_dimension"]), name="x_dimension"), row=1, col=2)
    fig.add_trace(go.Histogram(x=list(dt["y_dimension"]), name="y_dimension"), row=1, col=3)
    fig.add_trace(go.Histogram(x=list(dt["z_dimension"]), name="z_dimension"), row=1, col=4)
    fig.add_trace(go.Histogram(x=list(dt["cut"]), name="cut"), row=1, col=5)
    fig.add_trace(go.Histogram(x=list(dt["clarity"]), name="clarity"), row=2, col=1)
    fig.add_trace(go.Histogram(x=list(dt["color"]), name="color"), row=2, col=2)
    fig.add_trace(go.Histogram(x=list(dt["table"]), name="table"), row=2, col=3)
    fig.add_trace(go.Histogram(x=list(dt["depth"]), name="depth"), row=2, col=4)
    fig.add_trace(go.Histogram(x=list(dt["price"]), name="price"), row=2, col=5)
    fig.update_layout(margin={'l': 80, 'b': 80, 't': 10, 'r': 0})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
