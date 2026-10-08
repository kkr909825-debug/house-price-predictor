# House Price Predictor (Flask + scikit-learn)

Predicts house prices (INR) from city, size, road distance and width, bedrooms,
bathrooms, floors, property type, age, parking, furnishing, plus extra features
(balconies, metro distance, locality rating, facing, lift, security).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # trains the model on first run, then opens on http://localhost:5000
```

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Flask server, `/api/predict`, `/api/options`, `/api/metrics` |
| `train_model.py` | Trains RandomForest and GradientBoosting, keeps the better one |
| `data_generator.py` | Creates the synthetic dataset |
| `config.py` | Feature lists and input limits shared by training and the API |
| `templates/`, `static/` | Website (HTML, CSS, JS) |

## Use your own dataset
Create a CSV with these columns (see `config.py`) and a `price_inr` column, then:
```bash
python train_model.py --data my_data.csv
```

## API
`POST /api/predict` with JSON:
```json
{"city":"Pune","area_sqft":1200,"distance_main_road_m":100,"road_width_ft":30,
 "bedrooms":2,"bathrooms":2,"floors":1,"property_type":"Apartment","age_years":5,
 "parking":"Covered","furnishing":"Semi-furnished","facing":"East","balconies":1,
 "distance_metro_km":3,"locality_rating":3,"has_lift":1,"has_security":1}
```
Returns `price`, `low`, `high`, `price_per_sqft`.

## Deploy (Render / Railway / Heroku)
Push to GitHub and create a web service from the repo.
Build command: `pip install -r requirements.txt && python train_model.py`
Start command: `gunicorn app:app` (the `Procfile` already has this).
