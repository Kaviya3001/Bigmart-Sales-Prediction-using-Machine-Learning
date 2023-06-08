from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if len(request.form):
        Item_Weight = float(request.form['Item_Weight'])
        Item_Fat_Content = float(request.form['Item_Fat_Content'])
        Item_Visibility = float(request.form['Item_Visibility'])
        Item_Type = float(request.form['Item_Type'])
        Item_MRP = float(request.form['Item_MRP'])
        Outlet_Establishment_Year = float(request.form['Outlet_Establishment_Year'])
        Outlet_Size = float(request.form['Outlet_Size'])
        Outlet_Location_Type = float(request.form['Outlet_Location_Type'])
        Outlet_Type = float(request.form['Outlet_Type'])

        X = np.array([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year,
                     Outlet_Size, Outlet_Location_Type, Outlet_Type]])

        scaler_path = r'D:\DATA(D)\BigMart-Sales-Prediction-With-Deployment-main\models\sc.sav'

        sc = joblib.load(scaler_path)

        X_std = sc.transform(X)

        model_path = r'D:\DATA(D)\BigMart-Sales-Prediction-With-Deployment-main\models\lr.sav'

        model = joblib.load(model_path)

        Y_pred = model.predict(X_std)

        return jsonify({'Prediction': float(Y_pred)})
    else:
        return jsonify({})


if __name__ == "__main__":
    app.run(debug=True, port=9457)

