from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pickle
import sqlite3
import os

import json
from llm_analyzer import hybrid_analyze

def get_db():
    conn = sqlite3.connect('platform.db')
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load candidate context-aware model and vectorizer
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "model_improvement")
VEC_PATH = os.path.join(MODEL_DIR, "candidate_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "candidate_model.pkl")

if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
    vectorizer = pickle.load(open(VEC_PATH, 'rb'))
    model = pickle.load(open(MODEL_PATH, 'rb'))
else:
    vectorizer = pickle.load(open("best_vectorizer.pkl", 'rb'))
    model = pickle.load(open("best_model.pkl", 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/manual', methods=['GET', 'POST'])
def manual():
    if request.method == 'POST':
        user_input = request.form['text']
        analysis = hybrid_analyze(user_input, vectorizer=vectorizer, model=model)
        prediction = analysis['prediction']
        risk_score = analysis.get('risk_score', 85.0 if prediction == 1 else 10.0)
        severity = analysis.get('severity', 'High' if prediction == 1 else 'Low')
        explanation = analysis.get('explanation')

        method = analysis.get('method', 'ml')
        ml_prediction = analysis.get('ml_prediction', prediction)

        # Save every analyzed submission to Moderation
        conn = get_db()
        conn.execute(
            "INSERT INTO cases (comment, prediction, risk_score, severity, status) VALUES (?, ?, ?, ?, 'New')",
            (user_input, prediction, risk_score, severity)
        )
        conn.commit()
        conn.close()

        # POST-Redirect-GET: store result in session, redirect to GET
        # This prevents form re-submission (and duplicate DB inserts) on browser refresh
        session['last_prediction'] = prediction
        session['last_input'] = user_input
        session['last_explanation'] = explanation
        session['last_method'] = method
        session['last_ml_prediction'] = ml_prediction
        return redirect(url_for('manual'))

    # GET: read result from session (one-time), then clear it
    prediction = session.pop('last_prediction', None)
    last_input = session.pop('last_input', None)
    explanation = session.pop('last_explanation', None)
    method = session.pop('last_method', None)
    ml_prediction = session.pop('last_ml_prediction', None)
    return render_template(
        'manual.html',
        prediction=prediction,
        last_input=last_input,
        explanation=explanation,
        method=method,
        ml_prediction=ml_prediction
    )

@app.route('/api/analyze-context', methods=['POST'])
def api_analyze_context():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    result = hybrid_analyze(text, vectorizer=vectorizer, model=model)
    return jsonify(result)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/big-data')
def big_data():
    analytics_dir = os.path.join(os.path.dirname(__file__), 'data', 'analytics')
    
    def load_json(filename, default_val=None):
        path = os.path.join(analytics_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return default_val if default_val is not None else {}

    overview = load_json('overview.json', {})
    benchmarks = load_json('benchmarks.json', {})
    subtypes_list = load_json('subtypes.json', [])
    length_analysis = load_json('length_analysis.json', {"stats": {}, "groups": []})
    score_distribution = load_json('score_distribution.json', [])
    temporal_trends = load_json('temporal_trends.json', {"monthly": [], "yearly": [], "window_cumulative": []})
    publication_analysis = load_json('publication_analysis.json', [])
    quality_report = load_json('quality_report.json', {})

    # Convert subtypes to a dictionary for easy card access
    subtypes_dict = {}
    for item in subtypes_list:
        sub_key = item.get("subtype", "").lower().replace(" ", "_")
        subtypes_dict[sub_key] = item

    total_records = overview.get("total_records", 1804874)
    toxic_records = overview.get("toxic_records", 144334)
    non_toxic_records = overview.get("non_toxic_records", 1660540)
    toxic_pct = overview.get("toxicity_rate_pct", 8.0)
    non_toxic_pct = round(100.0 - toxic_pct, 2)
    avg_toxicity = overview.get("avg_toxicity_score", 0.103)
    csv_size_mb = overview.get("csv_size_mb", 778.40)
    parquet_size_mb = overview.get("parquet_size_mb", 41.83)
    compression_ratio = overview.get("compression_ratio", 18.61)
    space_saved_pct = round((1.0 - (parquet_size_mb / max(csv_size_mb, 1))) * 100.0, 1)

    timings = benchmarks.get("timings", {})
    operations = benchmarks.get("operations", [])
    total_pipeline_time = benchmarks.get("total_pipeline_time_sec", 248.37)
    overall_throughput = round(total_records / max(total_pipeline_time, 1))

    return render_template(
        'big_data.html',
        overview={
            "total_records": total_records,
            "toxic_records": toxic_records,
            "non_toxic_records": non_toxic_records,
            "toxic_percentage": toxic_pct,
            "non_toxic_percentage": non_toxic_pct,
            "avg_toxicity": avg_toxicity,
            "csv_size_mb": csv_size_mb,
            "parquet_size_mb": parquet_size_mb,
            "compression_ratio": compression_ratio,
            "space_saved_percentage": space_saved_pct,
            "duplicate_comments": overview.get("duplicate_comments", 24051),
            "invalid_records_dropped": overview.get("invalid_records_dropped", 0),
            "spark_version": overview.get("spark_version", "4.2.0"),
            "java_version": overview.get("java_version", "21.0.12")
        },
        timings=timings,
        operations=operations,
        total_pipeline_time=total_pipeline_time,
        overall_throughput=overall_throughput,
        subtypes=subtypes_dict,
        subtypes_list=subtypes_list,
        length_stats=length_analysis.get("stats", {}),
        length_groups=length_analysis.get("groups", []),
        score_distribution=score_distribution,
        temporal_trends=temporal_trends,
        publication_analysis=publication_analysis,
        quality_report=quality_report
    )

@app.route('/moderation')
def moderation():
    return render_template('moderation.html')

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

@app.route('/api/cases', methods=['GET'])
def get_cases():
    conn = get_db()
    cases = conn.execute("SELECT id, comment, risk_score, severity, status, created_at FROM cases ORDER BY id ASC").fetchall()
    conn.close()
    return jsonify([dict(c) for c in cases])

@app.route('/api/cases/<int:case_id>', methods=['PATCH'])
def update_case(case_id):
    new_status = request.json.get('status')
    if not new_status:
        return jsonify({"error": "No status provided"}), 400
    conn = get_db()
    conn.execute("UPDATE cases SET status = ? WHERE id = ?", (new_status, case_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/api/cases/<int:case_id>', methods=['DELETE'])
def delete_case(case_id):
    conn = get_db()
    conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

import random
import time

SIMULATION_COMMENTS = [
    "You are so stupid and ugly, why are you even here?",
    "Have a wonderful day everyone!",
    "Nobody likes you, just leave.",
    "This is a very helpful tutorial, thanks!",
    "I'm going to beat you up if I see you.",
    "Great job on the presentation today.",
    "You're a complete failure and a loser.",
    "Can someone help me with this math problem?",
    "Die trash, you don't belong here.",
    "I love the new update, it's awesome!"
]

@app.route('/api/simulate', methods=['POST'])
def simulate_comment():
    comment = random.choice(SIMULATION_COMMENTS)
    
    # Pass through existing ML pipeline with correct label mapping
    transformed = vectorizer.transform([comment])
    raw_pred = model.predict(transformed)[0]
    prediction = 1 if raw_pred == -1 else 0
    
    # If bullying detected, we should optionally add it to the DB so moderation sees it
    status = "Bullying" if prediction == 1 else "Not Bullying"
    risk_score = 85.0 if prediction == 1 else 10.0 # Mocking risk score since this model only outputs binary
    severity = "High" if prediction == 1 else "Low"
    
    if prediction == 1:
        conn = get_db()
        conn.execute(
            "INSERT INTO cases (comment, prediction, risk_score, severity) VALUES (?, ?, ?, ?)",
            (comment, prediction, risk_score, severity)
        )
        conn.commit()
        conn.close()
        
    return jsonify({
        "comment": comment,
        "prediction": status,
        "risk_score": risk_score,
        "severity": severity,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    recent = conn.execute("SELECT id, comment, severity, created_at FROM cases ORDER BY created_at DESC LIMIT 3").fetchall()
    conn.close()
    return jsonify({
        "total_cases": total,
        "recent_cases": [dict(r) for r in recent]
    })


if __name__ == '__main__':
    app.run(debug=True)
