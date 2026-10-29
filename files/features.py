"""Features available when a home is listed, for both competition tracks.

Never use sold_price, date_sold, days_on_market, or the external AVM here.
Off-market must never receive list_price.
"""
import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ("house_type_name", "house_style", "municipality", "community")

# Simple listing-description indicators. These are fixed before validation, so no
# vocabulary or statistics are learned from the future or the hidden test homes.
REMARK_PATTERNS = {
    "renovated": r"renovat|updated|modern|upgrad|remodel",
    "needs_work": r"needs work|fixer|as.is|tlc|original condition",
    "transit": r"subway|transit|t.tc|go station",
}


def square_feet(house_area):
    """Convert a single area or a range such as 900-1099 to square feet."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def _base_features(homes):
    """Make the listing-time features shared by both tracks."""
    x = pd.DataFrame(index=homes.index)
    x["square_feet"] = square_feet(homes["house_area"])
    for source, name in (("bedrooms", "bedrooms"), ("washrooms", "washrooms"),
                         ("parking_total", "parking"), ("garage", "garage"),
                         ("lot_front", "lot_front"), ("lot_depth", "lot_depth"),
                         ("latitude", "latitude"), ("longitude", "longitude")):
        x[name] = pd.to_numeric(homes[source], errors="coerce")
    x["lot_area"] = x["lot_front"] * x["lot_depth"]
    listed = pd.to_datetime(homes["date_listed"])
    x["months_since_2021"] = (listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    x["listing_month_sin"] = np.sin(2 * np.pi * listed.dt.month / 12)
    x["listing_month_cos"] = np.cos(2 * np.pi * listed.dt.month / 12)
    for name in CATEGORICAL_COLUMNS:
        x[name] = homes[name].fillna("Unknown").astype(str)
    remarks = homes["remarks"].fillna("").astype(str).str.lower()
    for name, pattern in REMARK_PATTERNS.items():
        x["remark_" + name] = remarks.str.contains(pattern, regex=True).astype(int)
    return x


def off_market_features(homes):
    """Add listing-time details useful when asking price is unavailable."""
    x = _base_features(homes)
    area_parts = homes["house_area"].astype(str).str.split("-", n=1, expand=True).reindex(columns=[0, 1])
    area_low = pd.to_numeric(area_parts[0], errors="coerce")
    area_high = pd.to_numeric(area_parts[1], errors="coerce").fillna(area_low)
    x["area_low"] = area_low
    x["area_width"] = area_high - area_low
    square_feet_safe = x["square_feet"].replace(0, np.nan)
    x["lot_to_house_area"] = x["lot_area"] / square_feet_safe
    x["bedrooms_per_1000sqft"] = 1000 * x["bedrooms"] / square_feet_safe
    x["washrooms_per_1000sqft"] = 1000 * x["washrooms"] / square_feet_safe
    x["washrooms_per_bedroom"] = x["washrooms"] / x["bedrooms"].replace(0, np.nan)
    x["garage_share_of_parking"] = x["garage"] / x["parking"].replace(0, np.nan)

    return _add_common_remark_details(x, homes)


def _add_common_remark_details(x, homes):
    """Add description signals frequent enough to use in both tracks."""
    remarks = homes["remarks"].fillna("").astype(str).str.lower()
    x["remark_words"] = remarks.str.count(r"\b\w+\b")
    patterns = {
        "basement": r"basement|lower level",
        "ravine": r"ravine",
        "view": r"unobstructed|panoramic|city view",
        "balcony": r"balcony|terrace",
        "locker": r"locker|storage unit",
        "move_in": r"move.in ready|turnkey",
    }
    for name, pattern in patterns.items():
        x["remark_" + name] = remarks.str.contains(pattern, regex=True).astype(int)
    return x


def list_aware_features(homes):
    """Add asking-price information available only in the list-aware track."""
    x = _base_features(homes)
    x["list_price"] = pd.to_numeric(homes["list_price"], errors="coerce")
    x["list_price_per_sqft"] = x["list_price"] / x["square_feet"].replace(0, np.nan)
    return _add_common_remark_details(x, homes)
