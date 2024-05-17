from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

df = pd.read_csv('../models/yelp_restaurant_review_labelled.csv')

categories = ["FOOD", "LOCATION", "ATOMSPHERE", "SERVICE", "PRICE", "MENU", "SPEED"]


def draw_graph(threshold = 0.1):
    random_entries = df.sample(1)
    for category in categories:
        usefulness = f'{category}_usefulness'
        score = f'{category}_score'
        random_entries.loc[random_entries[usefulness] < threshold, score] = 0
        random_entries.drop(columns=[usefulness], inplace=True)

    return [
        dbc.Card(
            [
                dbc.CardHeader(f'Text: {row["text"]}'),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            figure=px.bar(
                                x=categories,
                                y=[row[f'{category}_score'] for category in categories],
                                title='Scores',
                                labels={'x': 'Category', 'y': 'Score'},
                                range_y=[1, 5]
                            ),
                        )
                    ]
                )
            ]
        )
        for i, row in random_entries.iterrows()
    ]

external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = dbc.Container(
    [
        html.H1('Random Reviews'),
        dbc.Button('New Review', id='refresh', n_clicks=0, style={'margin-bottom': '10px'}),
        dbc.Row(
            children=draw_graph(),
            id='bar-graphs'
        )
    ]
)

@callback(
    Output('bar-graphs', 'children'),
    Input('refresh', 'n_clicks'),
)
def update_graphs(n_clicks):
    return draw_graph()


if __name__ == '__main__':
    app.run(debug=True)