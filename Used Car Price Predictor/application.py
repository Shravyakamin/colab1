from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

model=pickle.load(open("LinearRegressionModel.pkl",'rb'))
app=Flask(__name__)
car =pd.read_csv("Cleaned Car.csv")

@app.route('/',methods=['GET','POST'])
def index():
    city= sorted(car['city'].unique())
    car_models= sorted(car['car_name'].unique())
    year= sorted(car['year_of_manufacture'].unique(),reverse=True)
    fuel_type= sorted(car['fuel_type'].unique())
    return render_template('index.html',cities=city,carmodels=car_models,years=year,fuel_types=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    city=request.form.get('city')
    car_model=request.form.get('model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel')
    driven=int(request.form.get('kilo_driven'))
    print(city,car_model,year,fuel_type,driven)

    prediction=model.predict(pd.DataFrame([[car_model,driven,fuel_type,city,year]], columns=['car_name', 'kms_driven', 'fuel_type', 'city', 'year_of_manufacture']))
    
    return str(np.round(prediction[0],2))
    
   



if __name__ =="__main__":
    app.run(debug=True)