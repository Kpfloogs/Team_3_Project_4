import numpy as np
import sqlite3
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import pickle4 as pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def welcome():
    """List all available api routes."""
    return (
        f"Available Routes:<br/>"
        f"/api/v1.0/heart_stroke_data"
    )

def get_db_conn():
    conn = sqlite3.connect('heart_stroke.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/v1.0/heart_stroke_data')
@cross_origin()
def heart_stroke_data():
    conn = get_db_conn()
    posts = conn.execute('SELECT * FROM stroke').fetchall()
    conn.close()
    # posts = list(np.ravel(posts))
    data = []
        
    for post in posts:
        heart_stroke_data = {}
        heart_stroke_data["id"] = post[0]
        heart_stroke_data["gender"] = post[1]
        heart_stroke_data["age"] = post[2]
        heart_stroke_data["hypertension"] = post[3]
        heart_stroke_data["heart_disease"] = post[4]
        heart_stroke_data["ever_married"] = post[5]
        heart_stroke_data["work_type"] = post[6]
        heart_stroke_data["Residence_type"] = post[7]
        heart_stroke_data["avg_glucose_level"] = post[8]
        heart_stroke_data["bmi"] = post[9]
        heart_stroke_data["smoking_status"] = post[10]
        heart_stroke_data["stroke"] = post[11]

        data.append(heart_stroke_data)

    return jsonify(data)

    

if __name__ == '__main__':
    app.run(debug = True)