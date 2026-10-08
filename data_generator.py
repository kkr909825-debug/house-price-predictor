"""
Generates a realistic SYNTHETIC housing dataset (prices in INR).

Replace this with a real dataset (Kaggle, 99acres/MagicBricks scrape, etc.)
by passing --data your.csv to train_model.py. The CSV must contain the
columns listed in config.ALL_FEATURES plus `price_inr`.
"""
import numpy as np
import pandas as pd

from config import CITIES, PROPERTY_TYPES, FURNISHING, FACING, PARKING, TARGET

# Approx base price per sq ft (INR) for each city
CITY_RATE = {
    "Mumbai": 21000, "Delhi": 10500, "Bengaluru": 8500, "Hyderabad": 6800,
    "Pune": 7000, "Chennai": 7200, "Kolkata": 5600, "Ahmedabad": 4800,
    "Jaipur": 4200, "Lucknow": 3800,
}
TYPE_MULT = {"Apartment": 1.0, "Independent House": 1.12, "Villa": 1.35,
             "Builder Floor": 1.05, "Penthouse": 1.5}
FURNISH_MULT = {"Unfurnished": 1.0, "Semi-furnished": 1.04, "Furnished": 1.09}
PARKING_MULT = {"None": 0.96, "Open": 1.0, "Covered": 1.04}
FACING_MULT = {"North": 1.01, "South": 0.98, "East": 1.03, "West": 0.99}


def generate(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    city = rng.choice(CITIES, n)
    ptype = rng.choice(PROPERTY_TYPES, n, p=[0.52, 0.18, 0.08, 0.17, 0.05])
    furnishing = rng.choice(FURNISHING, n, p=[0.4, 0.4, 0.2])
    facing = rng.choice(FACING, n)
    parking = rng.choice(PARKING, n, p=[0.15, 0.35, 0.5])

    area = np.clip(rng.lognormal(7.05, 0.45, n), 350, 9000).round()
    bedrooms = np.clip(np.round(area / 550 + rng.normal(0, 0.6, n)), 1, 7).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 7).astype(int)
    floors = np.where(np.isin(ptype, ["Independent House", "Villa"]),
                      rng.integers(1, 4, n), 1)
    balconies = np.clip(rng.integers(0, 4, n), 0, bedrooms)
    age = np.clip(rng.gamma(2.2, 5.0, n), 0, 60).round()
    dist_road = np.clip(rng.exponential(120, n), 0, 1500).round()
    road_width = np.clip(rng.normal(30, 12, n), 10, 100).round()
    dist_metro = np.clip(rng.exponential(4.5, n), 0.2, 35).round(1)
    rating = np.clip(np.round(rng.normal(3.4, 1.0, n)), 1, 5).astype(int)
    lift = ((ptype == "Apartment") | (ptype == "Penthouse")) & (rng.random(n) < 0.8)
    security = rng.random(n) < 0.6

    rate = np.array([CITY_RATE[c] for c in city], dtype=float)
    mult = (
        np.array([TYPE_MULT[t] for t in ptype])
        * np.array([FURNISH_MULT[f] for f in furnishing])
        * np.array([PARKING_MULT[p] for p in parking])
        * np.array([FACING_MULT[f] for f in facing])
    )
    mult *= (1 - 0.007 * age).clip(0.65, 1)                  # depreciation
    mult *= 1 + 0.015 * (road_width - 30) / 10                # wider road = premium
    mult *= (1 - 0.00018 * dist_road).clip(0.85, 1)           # far from road = discount
    mult *= (1 - 0.012 * dist_metro).clip(0.7, 1)             # metro proximity
    mult *= 0.88 + 0.06 * rating                              # locality quality
    mult *= 1 + 0.025 * lift + 0.03 * security + 0.01 * balconies
    mult *= 1 + 0.01 * (bathrooms - bedrooms)
    mult *= (1.0 - 0.00003 * (area - 1200)).clip(0.8, 1.1)    # bulk discount per sqft

    price = rate * area * mult * rng.lognormal(0, 0.07, n)    # market noise

    return pd.DataFrame({
        "city": city, "area_sqft": area, "distance_main_road_m": dist_road,
        "road_width_ft": road_width, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "floors": floors, "property_type": ptype, "age_years": age,
        "parking": parking, "furnishing": furnishing, "facing": facing,
        "balconies": balconies, "distance_metro_km": dist_metro,
        "locality_rating": rating, "has_lift": lift.astype(int),
        "has_security": security.astype(int), TARGET: price.round(-3),
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/housing_data.csv", index=False)
    print(df.head(), f"\nSaved {len(df)} rows -> data/housing_data.csv")
