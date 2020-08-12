from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__, template_folder = "templates")

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/get_company_names', methods=['GET'])
def get_company_names():
    response = jsonify({
        'company': util.get_company_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/get_models_names', methods=['GET'])
def get_models_names():
    response = jsonify({
        'models': util.get_models_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/get_trans_names', methods=['GET'])
def get_trans_names():
    response = jsonify({
        'Transmission': util.get_trans_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/get_fuel_names', methods=['GET'])
def get_fuel_names():
    response = jsonify({
        'Fuel_Type': util.get_fuel_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/get_owner_names', methods=['GET'])
def get_owner_names():
    response = jsonify({
        'Owner_Type': util.get_owner_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
@app.route('/predict_car_price', methods=['POST'])
def predict_home_price():
    Power = float(request.form['Power'])
    Mileage = float(request.form['Mileage'])
    Company = request.form['Company']
    Model = request.form['Model']
    Fuel_Type = request.form['Fuel_Type']
    Owner_Type = request.form['Owner_Type']
    Transmission = request.form['Transmission']
    Year = int(request.form['Year'])
    Engine = int(request.form['Engine'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(Company,Model, Fuel_Type, Transmission,
                  Owner_Type, Year, Mileage, Engine, 
                  Power)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()