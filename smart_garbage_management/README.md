# EcoTrack — Smart Garbage Management System

A DWM lab mini-project built with Python Flask, SQLite, HTML, CSS, JavaScript and Chart.js.

## Features
- Dashboard KPIs: total bins, critical bins, average fill, logged waste collected
- Add, update, delete, search and filter bins
- Status thresholds: Empty (<20%), Normal (20–49%), Half Full (50–79%), Full (80–100%)
- Mark a bin collected; its fill level resets and an estimated collection weight is logged
- Analytics charts for bin fill levels and waste categories
- SQLite database created and seeded automatically

## Requirements
- Python 3.9 or newer
- Internet connection for Chart.js and Google Fonts CDN (dashboard and core forms still load without fonts; charts need Chart.js)

## Run on Windows
Open Command Prompt or VS Code terminal inside this folder:

```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

If `py` does not work, use `python` instead.

## Database
`garbage.db` is created automatically on first run. It stores bins and collection logs. Delete this file and restart the app to reset the demo data.

## DWM concepts demonstrated
- Data collection and persistence in SQLite
- Data cleaning/validation for fill levels
- Aggregation (`COUNT`, `SUM`, `GROUP BY`)
- Threshold-based classification and alerting
- Visualization of bin capacity and waste categories
- Actionable insights: prioritize bins at or above 80%

## Important note
The bin fill levels and initial collection weights are sample/demo data, not real IoT sensor readings. A future extension could send real sensor readings to the application through an API.


## Machine-learning forecast (demo)

EcoTrack includes a `RandomForestRegressor` in `ml_model.py`. It estimates each bin's fill percentage 24 hours ahead and displays the forecast on the dashboard. To keep the classroom demo self-contained, the model is trained on generated synthetic examples and assumed fill-growth rates; it is not trained on real sensor history and should not be described as a validated real-world predictor.

Install dependencies and run from the folder containing `app.py`:

```bash
python -m pip install -r requirements.txt
python app.py
```

For a real deployment, collect timestamped fill readings before/after collections, then train and evaluate the model on historical data using a chronological train/test split.
