
import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import root_mean_squared_error

# CHANGE THIS FILEPATH TO THE PROJECT FOLDER THAT CONTAINS THE TRAINING DATA
FILEPATH = "C:/Users/lavil/source/repos/LukVill/code/Python/MATH156/project/model_training_df.csv"


class la_restaurant_model():

    # input of parameters
    # order is specifically Food, Location, atmosphere, service, price, menu, speed, num reviews, num reviews, num price, num characters

    in_params = np.array([])
    prediction = 0
    feature_weights = 0
    error = 0
    x = 0
    y = 0

    def __init__(self, params):
        self.in_params = np.array(params)
        print("Starting up model...")

        # check if params is length 18
        if len(self.in_params) != 18:
            raise ValueError("18 parameters in model, received " + str(len(self.in_params)) + " parameters.")


        self.model = self.train_model()
        self.prediction = self.model.predict(self.in_params.reshape(1,-1))
        self.feature_weights = self.model.feature_importances_

        train_preds = self.model.predict(self.x)
        self.error = root_mean_squared_error(self.y, train_preds)
        
    def train_model(self):
        df = pl.read_csv(FILEPATH)
        self.x = df.select(pl.exclude("avg_rating"))
        self.y = df.select(pl.col("avg_rating"))
        tree_model = DecisionTreeRegressor(max_depth=3)
        return tree_model.fit(X=self.x,y=self.y)


    # make predicting function that takes in restaurant's traits
    # latitude, longitude, cuisine_count, number of reviews, average review length, food_score, food_usefulness, atmosphere_score, atmosphere_usefulness, service_score, 

    # print/export out the weights of the model


# test as singular script
# if __name__ == "__main__":
#     md = la_restaurant_model(np.random.random(18))
#     print(md.in_params)
#     # print(md.prediction)
#     # weights sum up to 1
#     print(md.feature_weights)
#     # ANALYSIS: review num, category num, price count, and menu score are significant
#     # with menu score being most important, followed by number of reviews
#     # IDEA: filter out variables to ones above 0, and train model to those features
#     # print(md.error)

