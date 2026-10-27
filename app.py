print("CivicSense AI is starting...")

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import sqlite3


# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# LOAD TRAINED ML MODEL
# --------------------------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ml",
    "civic_model.pkl"
)

model = joblib.load(model_path)

print("MODEL LOADED SUCCESSFULLY!")
# Load predictive waterlogging model
prediction_model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "predictiveml",
    "civic_prediction_model.pkl"
)

prediction_model = joblib.load(prediction_model_path)

print("PREDICTIVE MODEL LOADED SUCCESSFULLY!")

# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "civic_sense.db"
)


def init_db():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            problem_type TEXT,

            location_type TEXT,

            severity INTEGER,

            frequency INTEGER,

            affected_people INTEGER,

            duration_days INTEGER,

            previous_reports INTEGER,

            environmental_risk INTEGER,

            priority TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print("DATABASE READY!")


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

# ============================================
# PREDICTIVE WATERLOGGING RISK
# ============================================

@app.route("/predict-risk", methods=["POST"])
def predict_risk():

    data = request.json

    # Create dataframe
    new_data = pd.DataFrame([{

        "rainfall_forecast_mm":
            data["rainfall_forecast_mm"],

        "previous_waterlogging":
            data["previous_waterlogging"],

        "drain_condition":
            data["drain_condition"],

        "recent_complaints":
            data["recent_complaints"],

        "affected_population":
            data["affected_population"],

        "location_type":
            data["location_type"],

        "drainage_capacity":
            data["drainage_capacity"],

        "soil_absorption":
            data["soil_absorption"]

    }])


    # Convert location into numbers
    new_data = pd.get_dummies(
        new_data,
        columns=["location_type"]
    )


    # Match training features
    model_features = prediction_model.feature_names_in_

    new_data = new_data.reindex(
        columns=model_features,
        fill_value=False
    )


    # Make prediction
    prediction = prediction_model.predict(
        new_data
    )[0]


    # Get probability
    probability = prediction_model.predict_proba(
        new_data
    )[0][1]


    risk_percentage = round(
        probability * 100,
        2
    )


    print(
        "Waterlogging Risk:",
        risk_percentage,
        "%"
    )


    return jsonify({

        "prediction":
            "LIKELY"
            if prediction == 1
            else "NOT LIKELY",

        "risk_probability":
            risk_percentage

    })

    # Get data sent from frontend
    data = request.json


    # --------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------

    new_problem = pd.DataFrame([{

        "problem_type": data["problem_type"],

        "location_type": data["location_type"],

        "severity": data["severity"],

        "frequency": data["frequency"],

        "affected_people": data["affected_people"],

        "duration_days": data["duration_days"],

        "previous_reports": data["previous_reports"],

        "environmental_risk": data["environmental_risk"]

    }])


    # --------------------------------------------------
    # CONVERT TEXT TO NUMBERS
    # --------------------------------------------------

    new_problem = pd.get_dummies(
        new_problem,
        columns=[
            "problem_type",
            "location_type"
        ]
    )


    # --------------------------------------------------
    # MATCH TRAINING FEATURES
    # --------------------------------------------------

    model_features = model.feature_names_in_

    new_problem = new_problem.reindex(
        columns=model_features,
        fill_value=False
    )


    # --------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------

    prediction = model.predict(new_problem)[0]

    print("Predicted Priority:", prediction)


    # --------------------------------------------------
    # SAVE REPORT TO DATABASE
    # --------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        INSERT INTO reports (

            problem_type,

            location_type,

            severity,

            frequency,

            affected_people,

            duration_days,

            previous_reports,

            environmental_risk,

            priority

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        data["problem_type"],

        data["location_type"],

        data["severity"],

        data["frequency"],

        data["affected_people"],

        data["duration_days"],

        data["previous_reports"],

        data["environmental_risk"],

        prediction

    ))


    conn.commit()

    conn.close()


    print("REPORT SAVED TO DATABASE!")


    # --------------------------------------------------
    # SEND RESULT TO FRONTEND
    # --------------------------------------------------

    return jsonify({

        "priority": prediction

    })


# --------------------------------------------------
# START FLASK SERVER
# --------------------------------------------------
# --------------------------------------------------
# DASHBOARD API
# --------------------------------------------------

@app.route("/dashboard", methods=["GET"])
def dashboard():

    conn = sqlite3.connect(DB_PATH)

    # Total reports
    total_reports = conn.execute(
        "SELECT COUNT(*) FROM reports"
    ).fetchone()[0]

    # High priority
    high_priority = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE priority = 'High'"
    ).fetchone()[0]

    # Medium priority
    medium_priority = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE priority = 'Medium'"
    ).fetchone()[0]

    # Low priority
    low_priority = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE priority = 'Low'"
    ).fetchone()[0]

    # Recent reports
    recent_reports = conn.execute("""
        SELECT
            id,
            problem_type,
            location_type,
            priority,
            created_at
        FROM reports
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    reports = []

    for report in recent_reports:

        reports.append({
            "id": report[0],
            "problem_type": report[1],
            "location_type": report[2],
            "priority": report[3],
            "created_at": report[4]
        })

    return jsonify({

        "total_reports": total_reports,

        "high_priority": high_priority,

        "medium_priority": medium_priority,

        "low_priority": low_priority,

        "recent_reports": reports

    })
if __name__ == "__main__":

    init_db()

    app.run(debug=True)
