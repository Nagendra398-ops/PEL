"""Demo ML model for EcoTrack.

The training examples below are synthetic, generated for an academic demo. Replace
these examples with real timestamped bin readings before making real-world forecasts.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

WASTE_TYPES = {
    "Mixed": 0,
    "Organic": 1,
    "Plastic": 2,
    "Paper": 3,
    "Recyclable": 4,
    "Metal": 5,
    "Glass": 6,
}

# Assumed average fill growth in percentage points per hour (synthetic demo assumptions).
GROWTH_RATES = {
    "Mixed": 1.1,
    "Organic": 1.7,
    "Plastic": 0.45,
    "Paper": 0.7,
    "Recyclable": 0.5,
    "Metal": 0.25,
    "Glass": 0.2,
}


def _train_model():
    rng = np.random.default_rng(42)
    rows = []
    targets = []

    # Generate illustrative training examples across fill levels, waste types and forecast horizons.
    for _ in range(3500):
        fill = int(rng.integers(0, 101))
        waste_type = str(rng.choice(list(WASTE_TYPES.keys())))
        hours_ahead = int(rng.integers(1, 49))
        noise = float(rng.normal(0, 4))
        future_fill = float(np.clip(
            fill + GROWTH_RATES[waste_type] * hours_ahead + noise, 0, 100
        ))
        rows.append([fill, WASTE_TYPES[waste_type], hours_ahead])
        targets.append(future_fill)

    model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3)
    model.fit(np.asarray(rows), np.asarray(targets))
    return model


_model = _train_model()


def predict_fill(current_fill, waste_type, hours_ahead=24):
    """Return estimated future fill percentage, clamped to 0..100."""
    type_code = WASTE_TYPES.get(waste_type, WASTE_TYPES["Mixed"])
    prediction = _model.predict([[float(current_fill), type_code, float(hours_ahead)]])[0]
    return max(0, min(100, int(round(prediction))))
