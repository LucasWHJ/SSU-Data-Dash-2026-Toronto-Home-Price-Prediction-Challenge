"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import numpy as np
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]

REMARKS_KEYWORDS = ["renovated", "upgrade", "luxury", "finished basement",
    "hardwood", "granite", "quartz", "stainless", "pot light",
    "custom", "pool", "ravine", "view", "balcony", "terrace",
    "spacious", "as-is", "fixer", "investor", "tlc"]

def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    area = square_feet(homes["house_area"])
    X["square_feet"] =  area
    X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["beds_per_wash"] = homes["bedrooms"] / homes["washrooms"].replace(0, np.nan)
    X["sqft_per_room"] = area / (homes["bedrooms"] + homes["washrooms"]).replace(0, np.nan)

    lot_front = pd.to_numeric(homes["lot_front"], errors="coerce").fillna(0)
    lot_depth = pd.to_numeric(homes["lot_depth"], errors="coerce").fillna(0)
    X["lot_front"] = lot_front
    X["lot_depth"] = lot_depth
    X["lot_area"] = lot_front * lot_depth
    X["lot_aspect_ratio"] = lot_depth / lot_front.replace(0, np.nan)

    X["parking"] = homes["parking_total"]
    X["garage"] = pd.to_numeric(homes["garage"], errors="coerce").fillna(0)
    lat = homes["latitude"]
    lon = homes["longitude"]
    X["latitude"] = lat
    X["longitude"] = lon
    X["dist_to_center"] = np.sqrt(((lon - lon.median()) * 0.72) ** 2 + (lat - lat.median()) ** 2)

    # Months since January 2021, so the model can follow prices over time.
    date_listed = pd.to_datetime(homes["date_listed"])
    X["months"] = (date_listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    X["list_month"] = date_listed.dt.month
    X["list_quarter"] = date_listed.dt.quarter

    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X

    for col in ["municipality", "community", "house_style"]:
        if col in homes.columns:
            freq = homes[col].value_counts(normalize=True)
            X[f"{col}_freq"] = homes[col].map(freq).fillna(0)

    if "remarks" in homes.columns:
        remarks_lower = homes["remarks"].astype(str).str.lower()
        for kw in REMARKS_KEYWORDS:
            col_name = "has_" + kw.replace(" ", "_").replace("-", "_")
            X[col_name] = remarks_lower.str.contains(kw, regex=False).astype(int)

    return X

def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X
