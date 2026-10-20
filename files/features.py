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
    "luxury": r"luxur|premium|high.end|custom built",
    "waterfront": r"waterfront|lake view|water view",
    "transit": r"subway|transit|t.tc|go station",
    "new_build": r"new build|newly built|brand new",
}


def square_feet(house_area):
    """Convert a single area or a range such as 900-1099 to square feet."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes):
    """Make only features known at listing time, without asking price."""
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


def list_aware_features(homes):
    """Add asking-price information available only in the list-aware track."""
    x = off_market_features(homes)
    x["list_price"] = pd.to_numeric(homes["list_price"], errors="coerce")
    x["list_price_per_sqft"] = x["list_price"] / x["square_feet"].replace(0, np.nan)
    return x
