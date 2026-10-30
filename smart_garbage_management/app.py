
import os
import json
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")
NEW_BINS_PATH = os.path.join(DATA_DIR, "new_bins.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "overflow_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# The trained model uses these exact features.
FEATURE_COLUMNS = [
    "bin_type",
    "fill_level_percent",
    "battery_level_percent",
    "hour",
    "day_of_week",
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_new_bins():
    """Load bins previously added through the dashboard."""
    if not os.path.exists(NEW_BINS_PATH):
        return []

    try:
        with open(NEW_BINS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        app.logger.exception("Could not read new_bins.json")
        return []


def save_new_bins(bins):
    with open(NEW_BINS_PATH, "w", encoding="utf-8") as file:
        json.dump(bins, file, indent=2, ensure_ascii=False)


def is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def get_bin_status(level):
    level = float(level)
    if level >= 80:
        return "Full"
    elif level >= 50:
        return "Half Full"
    elif level >= 20:
        return "Normal"
    return "Empty"


def load_test_records():
    """Read only held-out test records; never load demo/sample bins."""
    if not os.path.exists(TEST_DATA_PATH):
        return [], "Test data not found. Run python train_model.py first."

    try:
        df = pd.read_csv(TEST_DATA_PATH)
        required = [
            "bin_id",
            "bin_type",
            "fill_level_percent",
            "battery_level_percent",
            "overflow_alert",
            "predicted_overflow_alert",
            "prediction_correct",
        ]

        missing = [col for col in required if col not in df.columns]
        if missing:
            return [], "Missing columns in test_data.csv: " + ", ".join(missing)

        records = df.fillna("").to_dict(orient="records")

        for record in records:
            fill = pd.to_numeric(
                record.get("fill_level_percent", 0), errors="coerce"
            )
            record["fill_level_percent"] = (
                0.0 if pd.isna(fill) else float(fill)
            )

            battery = pd.to_numeric(
                record.get("battery_level_percent", ""), errors="coerce"
            )
            record["battery_display"] = (
                "N/A" if pd.isna(battery) else f"{float(battery):g}%"
            )

            actual = is_true(record.get("overflow_alert", False))
            predicted = is_true(
                record.get("predicted_overflow_alert", False)
            )

            record["actual_display"] = "Yes" if actual else "No"
            record["predicted_display"] = "Yes" if predicted else "No"
            record["result_display"] = (
                "Correct" if actual == predicted else "Incorrect"
            )
            record["result_class"] = (
                "correct" if actual == predicted else "incorrect"
            )

            city = str(record.get("address_city", "")).strip()
            record["city_display"] = city or "Unknown"

            street = str(record.get("address_street", "")).strip()
            record["location_display"] = ", ".join(
                part for part in (street, city) if part
            ) or "Location unavailable"

            record["status"] = get_bin_status(
                record["fill_level_percent"]
            )

        return records, None

    except Exception:
        app.logger.exception("Could not load test records")
        return [], "Could not read test_data.csv. Check the file format."


def get_all_cities(test_records, new_bins):
    cities = set()

    for record in test_records:
        city = str(record.get("city_display", "")).strip()
        if city:
            cities.add(city)

    for bin_record in new_bins:
        city = str(bin_record.get("city", "")).strip()
        if city:
            cities.add(city)

    return sorted(cities, key=str.lower)


@app.route("/", methods=["GET"])
def index():
    test_records, test_error = load_test_records()
    new_bins = load_new_bins()

    # Prepare test-record monitoring values.
    test_bins = []
    for record in test_records:
        test_bins.append({
            "bin_code": str(record.get("bin_id", "N/A")),
            "city": record.get("city_display", "Unknown"),
            "location": record.get("location_display", "N/A"),
            "waste_type": str(record.get("bin_type", "N/A")),
            "fill_level": record.get("fill_level_percent", 0),
            "battery": record.get("battery_display", "N/A"),
            "status": record.get("status", "Unknown"),
            "actual": record.get("actual_display", "N/A"),
            "predicted": record.get("predicted_display", "N/A"),
            "result": record.get("result_display", "N/A"),
            "result_class": record.get("result_class", ""),
            "timestamp": str(record.get("timestamp", "N/A")),
        })

    # New bins are separate from the held-out test records.
    for bin_record in new_bins:
        bin_record["status"] = get_bin_status(
            bin_record.get("fill_level_percent", 0)
        )

    total_test_records = len(test_bins)
    critical_test_records = sum(
        1 for record in test_bins if record["fill_level"] >= 80
    )
    average_fill = round(
        sum(record["fill_level"] for record in test_bins)
        / total_test_records,
        1,
    ) if total_test_records else 0

    correct_count = sum(
        1 for record in test_bins if record["result"] == "Correct"
    )
    test_accuracy = round(
        correct_count * 100 / total_test_records, 1
    ) if total_test_records else 0

    full_count = sum(
        1 for record in test_bins if record["fill_level"] >= 80
    )
    half_count = sum(
        1 for record in test_bins
        if 50 <= record["fill_level"] < 80
    )
    normal_count = sum(
        1 for record in test_bins
        if 20 <= record["fill_level"] < 50
    )
    empty_count = sum(
        1 for record in test_bins if record["fill_level"] < 20
    )

    return render_template(
        "index.html",
        test_bins=test_bins,
        new_bins=new_bins,
        cities=get_all_cities(test_records, new_bins),
        test_error=test_error,
        total_test_records=total_test_records,
        critical_test_records=critical_test_records,
        average_fill=average_fill,
        test_accuracy=test_accuracy,
        full_count=full_count,
        half_count=half_count,
        normal_count=normal_count,
        empty_count=empty_count,
    )


@app.route("/add-bin", methods=["POST"])
def add_bin():
    bin_id = request.form.get("bin_id", "").strip()
    city = request.form.get("city", "").strip()
    location = request.form.get("location", "").strip()
    bin_type = request.form.get("bin_type", "").strip()

    try:
        fill_level = float(request.form.get("fill_level_percent", ""))
        battery_level = float(
            request.form.get("battery_level_percent", "")
        )
    except (TypeError, ValueError):
        flash("Fill level aur battery level valid numbers hone chahiye.", "error")
        return redirect(url_for("index"))

    if not all([bin_id, city, bin_type]):
        flash("Bin ID, city aur bin type required hain.", "error")
        return redirect(url_for("index"))

    if not 0 <= fill_level <= 100:
        flash("Fill level 0 se 100 ke beech hona chahiye.", "error")
        return redirect(url_for("index"))

    if not 0 <= battery_level <= 100:
        flash("Battery level 0 se 100 ke beech hona chahiye.", "error")
        return redirect(url_for("index"))

    existing_bins = load_new_bins()
    if any(
        str(item.get("bin_id", "")).lower() == bin_id.lower()
        for item in existing_bins
    ):
        flash("Is ID ka new bin pehle se exist karta hai.", "error")
        return redirect(url_for("index"))

    # Load the same model that was trained by train_model.py.
    try:
        model = load_model()
    except Exception:
        app.logger.exception("Could not load trained model")
        model = None

    if model is None:
        flash("Trained model nahi mila. Pehle python train_model.py run karo.", "error")
        return redirect(url_for("index"))

    now = datetime.now(timezone.utc)

    # Feature names and order must match training.
    model_input = pd.DataFrame([{
        "bin_type": bin_type,
        "fill_level_percent": fill_level,
        "battery_level_percent": battery_level,
        "hour": now.hour,
        "day_of_week": now.weekday(),
    }], columns=FEATURE_COLUMNS)

    try:
        prediction = bool(model.predict(model_input)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            confidence = round(float(max(probabilities)) * 100, 1)

    except Exception:
        app.logger.exception("Model prediction failed")
        flash("Prediction nahi ho paayi. Model/features check karo.", "error")
        return redirect(url_for("index"))

    new_record = {
        "record_id": str(uuid.uuid4()),
        "bin_id": bin_id,
        "city": city,
        "location": location or "Not provided",
        "bin_type": bin_type,
        "fill_level_percent": fill_level,
        "battery_level_percent": battery_level,
        "timestamp": now.isoformat(),
        "predicted_overflow_alert": prediction,
        "predicted_display": "Yes" if prediction else "No",
        "prediction_confidence_percent": confidence,
        "status": get_bin_status(fill_level),
        # Actual label is unknown for a newly added bin.
        "actual_overflow_alert": None,
    }

    existing_bins.append(new_record)
    save_new_bins(existing_bins)

    flash(
        f"Bin {bin_id} added. Predicted overflow: "
        f"{new_record['predicted_display']}.",
        "success",
    )
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
