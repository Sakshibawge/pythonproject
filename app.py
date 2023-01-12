from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('heart.pk1','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Word"

@app.route('/predict',methods=['POST'])
def predict():
    Age = request.form.get('Age')
    Gender = request.form.get('Gender')
    Check_pain = request.form.get('Chest_pain')
    Blood_Suger = request.form.get('Blood_Sugar')
    Resting_Electrogram = request.form.get('Resting_Electrogram')
    Exercise_Induced =request.form.get('Exercise_Induced')
    Slope = request.form.get('Slope')
    Number_of_major = request.form.get('Number_of_major')
    Thal = request.form.get('Thal')
    Rest_Blood_Pressure = request.form.get('Rest_Blood_Pressure')
    Serum_Cholesterol = request.form.get('Serum_Cholesterol')
    Maximum_Heart_Rate_Achieved = request.form.get('Maximum_Heart_Rate_Achieved')
    ST_Depression_Induced_by_Exercise = request.form.get('ST_Depression_Induced_by_Exercise')

    input_query = np.array([[Age,Gender,Check_pain,Blood_Suger,Resting_Electrogram,Exercise_Induced,Slope,Number_of_major,Thal,Rest_Blood_Pressure,Serum_Cholesterol,Maximum_Heart_Rate_Achieved,ST_Depression_Induced_by_Exercise]])

    result = model.predict(input_query)[0]


    return jsonify({'disease':str(result)})

if __name__== '__main__':
    app.run(debug=True)
