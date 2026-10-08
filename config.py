"""Shared constants: feature definitions used by training AND the web app."""

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]
PROPERTY_TYPES = ["Apartment", "Independent House", "Villa", "Builder Floor", "Penthouse"]
FURNISHING = ["Unfurnished", "Semi-furnished", "Furnished"]
FACING = ["North", "South", "East", "West"]
PARKING = ["None", "Open", "Covered"]

NUMERIC_FEATURES = [
    "area_sqft",
    "distance_main_road_m",
    "road_width_ft",
    "bedrooms",
    "bathrooms",
    "floors",
    "age_years",
    "balconies",
    "distance_metro_km",
    "locality_rating",
    "has_lift",
    "has_security",
]
CATEGORICAL_FEATURES = ["city", "property_type", "parking", "furnishing", "facing"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price_inr"

# (min, max) accepted by the API for numeric inputs
NUMERIC_LIMITS = {
    "area_sqft": (200, 20000),
    "distance_main_road_m": (0, 5000),
    "road_width_ft": (6, 200),
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
    "floors": (1, 10),
    "age_years": (0, 100),
    "balconies": (0, 6),
    "distance_metro_km": (0, 60),
    "locality_rating": (1, 5),
    "has_lift": (0, 1),
    "has_security": (0, 1),
}
