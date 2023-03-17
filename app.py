import numpy as np
import sqlite3
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def welcome():
    """List all available api routes."""
    # return ("hello world")
    return render_template('index.html')


def get_db_conn():
    conn = sqlite3.connect('brain_stroke.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/v1.0/brain_stroke_data')
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

@app.route('/data')
def data_page():
    return render_template('data.html')

@app.route('/predictor')
def predict_page():
    return render_template('predictor.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        if gender == 'Male':
            gender_male = 1
            gender_female = 0
        else:
            gender_male = 0
            gender_female = 1

        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heartdisease = int(request.form['heartdisease'])
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])

        smoking_status = request.form['smoke']

        if smoking_status == 'formerly_smoked':
            smoking_status_Unkown = 0
            smoking_status_formerly = 1
            smoking_status_never = 0
            smoking_status_smokes = 0

        elif smoking_status == "never_smoked":
            smoking_status_Unkown = 0
            smoking_status_formerly = 0
            smoking_status_never = 1
            smoking_status_smokes = 0

        elif smoking_status == "smokes":
            smoking_status_Unkown = 0
            smoking_status_formerly = 0
            smoking_status_never = 0
            smoking_status_smokes = 1

        else:
            smoking_status_Unkown = 1
            smoking_status_formerly = 0
            smoking_status_never = 0
            smoking_status_smokes = 0

        final_features = np.array([[gender_male, gender_female, age, hypertension, heartdisease, glucose, bmi, smoking_status_Unkown, 
                                    smoking_status_formerly, smoking_status_never, smoking_status_smokes]])
        
        prediction = model.predict(final_features)

        return render_template('predictor.html', prediction=prediction)
    

if __name__ == '__main__':
    app.run(debug = True)