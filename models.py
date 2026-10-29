"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for 'off_market' or 'list_aware'."""
    if track == "list_aware":
        return ListAwareDeltaModel(features.list_aware_features)
    return FeatureModel(features.off_market_features)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class ListAwareDeltaModel:
    """List-aware 专用模型：建模 ln(sold_price) - ln(list_price) 的对数溢价率。"""

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42
        )

    def fit(self, train):
        # 过滤可能存在缺失或异常的挂牌价
        valid = train.dropna(subset=["list_price", "sold_price"])
        valid = valid[(valid["list_price"] > 0) & (valid["sold_price"] > 0)]

        X = self.make_features(valid)
        # 核心：目标是 log(sold_price / list_price)
        y = np.log(valid["sold_price"]) - np.log(valid["list_price"])
        self.model.fit(X, y)
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        # 预测相对溢价率，再乘以挂牌价还原出预测金额
        delta_pred = self.model.predict(X)
        return homes["list_price"].to_numpy(dtype=float) * np.exp(delta_pred)

class FeatureModel:
    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            random_state=42
        )

    def fit(self, train):
        valid = train.dropna(subset=["sold_price"])
        valid = valid[valid["sold_price"] > 0]
        X = self.make_features(valid)
        self.model.fit(X, np.log(valid["sold_price"]))
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        return np.exp(self.model.predict(X))
