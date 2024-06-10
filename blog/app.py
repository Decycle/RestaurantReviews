from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_latex as dl
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import numpy as np

# from components import Title
import time

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("sketchy")
external_stylesheets = [
    dbc.themes.SKETCHY,
    dbc.icons.FONT_AWESOME,
    dbc_css,
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.title = "Restaurant Reviews"

df = pd.read_csv("../models/yelp_restaurant_review_labelled.csv")

categories = [
    "FOOD",
    "LOCATION",
    "ATOMSPHERE",
    "SERVICE",
    "PRICE",
    "MENU",
    "SPEED",
]


def draw_graph(threshold=0.1):
    random_entry = df.sample()
    for category in categories:
        usefulness = f"{category}_usefulness"
        score = f"{category}_score"
        random_entry.loc[random_entry[usefulness] < threshold, score] = 0
        random_entry.drop(columns=[usefulness], inplace=True)
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H4("Review:", className="card-title"),
                    random_entry["text"].values[0],
                ],
            ),
            dbc.CardBody(
                [
                    dcc.Graph(
                        figure=px.bar(
                            x=categories,
                            y=[
                                random_entry[f"{category}_score"].values[0]
                                for category in categories
                            ],
                            title="Scores",
                            labels={"x": "Category", "y": "Score"},
                            range_y=[1, 5],
                        ),
                    )
                ]
            ),
        ]
    )


example_logits = np.array([7, 8, 10, 12, 9, 3, 11])
example_probs = np.exp(example_logits) / np.sum(np.exp(example_logits))

app.layout = dbc.Container(
    [
        html.H1("How we used Large Language Models to Rate Restaurant Reviews"),
        # by Dennis Yang
        html.H2("By Dennis Yang", className="fw-bold text-muted text-end"),
        html.H2("Sources", className="fw-bold"),
        # subtitle
        html.Span(
            """
            The code of this project is hosted at
            """
        ),
        html.A(
            "https://github.com/Decycle/RestaurantReviews",
            href="https://github.com/Decycle/RestaurantReviews",
        ),
        html.H2(
            [
                "Large Language Models",
                html.I(className="fa-solid fa-screwdriver-wrench fs-2"),
            ],
            className="fw-bold mt-2",
        ),
        html.P(
            """
            With the advancement of machine learning transformers like large language models, we can now analyze restaurant reviews in ways previously impossible. In this project, we will use a large language model to categorize and rate restaurant reviews, aiming to predict actual ratings based on category-specific review analysis.
        """,
            # higlighted text
            className="text-primary",
        ),
        html.H3("Prompting", className="fw-bold"),
        html.P(
            """
        So let's do this! Consider the following prompt:
        """
        ),
        # prompt of 3 rows, each start with emoji as the speaker, follwed by what they say
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.I(className="fas fa-user fs-1 text-start col-1"),
                        dcc.Markdown(
                            """Rate the following restaurant review in the category of **{category}** from 1 to 5 where 1 means the worst possible and 5 means the best in their life. Only rate how good the **{category}** is. Do not pay attention to other factors. If the category **{category}** is not mentioned in the review, output "*NOT MENTIONED*" instead. Review: "**{review}**" """,
                            className="text-end col-11",
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        html.I(className="fas fa-robot fs-1 text-start col-6"),
                        dcc.Markdown(
                            """The rating of **{category}** is: """,
                            className="text-end col-6",
                        ),
                    ],
                    className="mb-4",
                ),
            ],
        ),
        dcc.Markdown(
            """
            The **{category}** and **{review}** are placeholders for the actual category and review. For each review, we will pass the prompt to the model once per every category and collect the output.
            """
        ),
        html.P(
            """
            But what does the model really output? If you have been using large language models like Chat-GPT, you may be familiar with the block of text that the model outputs (and be annoyed by how word it is). But for our purposes, we only need the model to generate a single token. This will massively speed up the model because it is not spending 2 minutes writing up a whole paragraph.
            """
        ),
        html.P(
            """
            The output of the model may look something like this:
            """
        ),
        # example of model output
        # bar graph
        dcc.Graph(
            figure=px.bar(
                x=["1", "2", "3", "4", "5", "NOT", "[OTHER TOKENS]"],
                y=example_logits,
                title="Model Output",
                labels={"x": "Rating", "y": "Logits"},
            ),
        ),
        dcc.Markdown(
            """
            Each logit represents the log of the probability of the model outputting that token. In other words, the probability distribution is the *softmax* of the logits!
            """
        ),
        html.P(
            """
            The actual probabilities looks something like this:
            """
        ),
        dcc.Graph(
            figure=px.bar(
                x=["1", "2", "3", "4", "5", "NOT", "[OTHER TOKENS]"],
                y=example_probs,
                title="Model Output",
                labels={"x": "Rating", "y": "Probability"},
            ),
        ),
        html.P(
            """
            In this case, the model is really confident that the review should be rated 4 stars. Intuitively, we should also design the scoring mechanism such that the score is around 4.
            """
        ),
        html.H3("Scoring Mechanism", className="fw-bold"),
        dl.DashLatex(
            r"""
            The scoring mechanism we have chosen is really simple. It is simply a weighted average of all the ratings' probabilities.
            Let the probabilities of the model outputting a rating $i$ be $p_i$, then the score $s$ is simply
            """,
        ),
        dl.DashLatex(
            r"$$s = \frac{\sum_{i=1}^5 i \cdot p_i}{\sum_{i=1}^5 p_i}$$",
            displayMode=True,
        ),
        html.P("""for that category."""),
        dcc.Markdown(
            [
                """
            In the prompt you may have noticed that we specifically told the model to output *NOT MENTIONED* if the category is not mentioned in the review. Since we only let the model to generate a single token, we can collect the probability of *NOT* and use it to determine the **usefulness** of the model's prediction
            """
            ]
        ),
        dl.DashLatex(
            r"""
            $$u = \frac{\sum_{i=1}^5 p_i}{p_n + \sum_{i=1}^5 p_i}$$
            """,
            displayMode=True,
        ),
        dl.DashLatex(
            """
            where $p_n$ is the probability of the model outputting NOT.
            """
        ),
        dcc.Markdown(
            """
            We can then generate a mask to filtering out scores with low usefulness, and use the final scores to predict the actual ratings!
            """
        ),
        html.P(
            "I have included an interactive demo of some processed data. Click the button below to see a new example from the training data:"
        ),
        dbc.Button(
            [
                html.I(className="fas fa-sync-alt"),
                " New Review",
            ],
            id="refresh",
            n_clicks=0,
            className="mb-4",
        ),
        dbc.Col(
            children=[
                draw_graph(),
            ],
            id="bar-graphs",
        ),
        dcc.Markdown(
            """
            The bar graph above shows the scores of the review in each category. If a score is not shown, that means its usefulness is below 0.5 and is filtered out.
            """
        ),
        dcc.Markdown(
            """
            And that's it! We have successfully used a large language model to get categorical scores for the text part of the review. We can use these scores however we want. In the final report, my team members used these scores and other features to train various different models and reached a final RMSE of 0.28 on the test set, which is pretty good!
            """
        ),
    ],
    className="app-body dbc",
    fluid=True,
)


@callback(
    Output("bar-graphs", "children"),
    Input("refresh", "n_clicks"),
)
def update_graphs(n_clicks):
    return draw_graph()


if __name__ == "__main__":
    app.run(debug=True)
