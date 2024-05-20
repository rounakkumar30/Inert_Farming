# importing all the required python modules
from bs4 import BeautifulSoup
from utils.disease import disease_dic
from utils.model import ResNet9
from PIL import Image
import xgboost
from torchvision import transforms
import torch
import io
from datetime import datetime
import urllib.parse
from utils.fertilizer import fertilizer_dic
from flask import Flask, render_template, request, redirect
from datetime import datetime
from mongoengine import Document, StringField, DateTimeField
from utils.crop import crop
import plotly.express as px
import plotly
import json
from flask import Flask, flash, redirect, request, render_template, session, url_for
from markupsafe import Markup
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn import datasets
from flask_mongoengine import MongoEngine
from werkzeug.security import generate_password_hash, check_password_hash, gen_salt
app = Flask(__name__)


# Mongodb Database Connection Use localHost
name = 'farming'
username = urllib.parse.quote_plus('name')
app.secret_key = 'aifarming'
app.config['MONGODB_SETTINGS'] = {
    'db': name,
    'host': 'mongodb://localhost:27017/'+name,
# database connection name
}
db = MongoEngine()
db1 = MongoEngine()
db.init_app(app)
db1.init_app(app)


class users(db.Document):
    username = db.StringField()
    email = db.StringField()
    phone = db.StringField()
    profession = db.StringField()
    password = db.StringField()
    rpassword = db.StringField()
    registered_Date = db.DateTimeField(default=datetime.now)

class Market(Document):
    fname = StringField()
    lname = StringField()
    email = StringField()
    phone = StringField()
    address = StringField()
    croptype = StringField()
    quantity = StringField()
    cropname = StringField()
    msp = StringField()
    registered_Date = DateTimeField(default=datetime.now)


class ContactForm(db.Document):
    name = db.StringField(required=True)
    email = db.EmailField(required=True)
    message = db.StringField(required=True)
  

# Route to render the form for selling crops
@app.route('/sell', methods=['GET'])
def sell():
    return render_template('sell.html')

# Route to handle the form submission
@app.route('/submit_sell_form', methods=['POST'])
def submit_sell_form():
    if request.method == 'POST':
        try:
            # Get form data
            fname = request.form['fname']
            lname = request.form['lname']
            email = request.form['email']
            phone = request.form['phone']
            address = request.form['add']
            croptype = request.form['ctype']
            quantity = request.form['Quantity']
            cropname = request.form['cname']
            msp = request.form['msp']

            # Store form data in MongoDB
            market = Market(
                fname=fname,
                lname=lname,
                email=email,
                phone=phone,
                address=address,
                croptype=croptype,
                quantity=quantity,
                cropname=cropname,
                msp=msp
            )
            market.save()

            # Redirect to a success page or do something else
            return 'Form submitted successfully!'

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return 'Invalid request'

#Contact
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


#Submit Function
@app.route('/submit_contact_form', methods=['POST'])
def submit_contact_form():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Create a new ContactForm document and save it to MongoDB
        contact_form = ContactForm(name=name, email=email, message=message)
        contact_form.save()

        return 'Form submitted successfully!'

    return 'Invalid request'
# Login Function


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        _username = request.form['email']
        _password = request.form['password']
        users1 = users.objects(email=_username).count()
        if users1 > 0:
            users_response = users.objects(email=_username).first()
            password = users_response['password']

            if check_password_hash(password, _password):
                session['logged_in'] = users_response['username']
                flash('You were logged In')
                return redirect(url_for('home'))
            else:
                error = "Invalid Login / Check Your Username And Password"
                return render_template('login.html', errormsg=error)
        else:
            error = "No User Exists"
            return render_template('login.html', errormsg=error)
    return render_template('login.html')


# Signup Function
@app.route('/signup', methods=['GET', 'POST'])
def register():

    today = datetime.today()

    if request.method == 'POST':
        _username = request.form['uname']
        _email = request.form['email']
        _phone = request.form['phone']
        _profession = request.form['phone']
        _password = request.form['password']
        _rpassword = request.form['rpassword']
        if _email and _username and _password:
            hashed_password = generate_password_hash(_password)
            users1 = users.objects(email=_email)
            if not users1:
                usersave = users(username=_username, email=_email, profession=_profession, phone=_phone,
                                password=hashed_password, rpassword=hashed_password, registered_Date=today)
                usersave.save()
                msg = '{"html":"OKay you have registered"}'
                msghtml = json.loads(msg)
                return msghtml["html"] and redirect('/login')
            else:
                msg = f"It seems that {_email} You have already Registered"
            
                return render_template('signup.html',msg=msg)
        else:
            msg="Please enter email address & required details"
            render_template("signup.html", msg=msg)
    return render_template("signup.html")


# logout Function
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were Logout Out Sucessfully')

    return redirect('/')


# importing and Loading Pickle File
model = pickle.load(open(r'crops.pkl', 'rb'))
fert = pickle.load(open(r'AIFarming-Deepcoders-main\fertilizer.pkl', 'rb'))


# API Based Waether data

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = '3b0124dbadcdbcf295fd8c009f8efc0c'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

# importing some information of crops


basePrice = {
    "paddy": 1940,
    "arhar": 6300,
    "bajra": 1250,
    "barley": 1650,
    "copra": 9920,
    "cotton": 3600,
    "sesamum": 6850,
    "gram": 4620,
    "groundnut": 5275,
    "jowar": 2430,
    "maize": 1850,
    "masoor": 4800,
    "moong": 7196,
    "nigerseed": 5500,
    "pulses": 4800,
    "ragi": 3295,
    "rape": 2500,
    "jute": 4160,
    "safflower": 5325,
    "soyabean": 3880,
    "sugarcane": 290,
    "sunflower": 5885,
    "urad": 6000,
    "wheat": 1975
}

a = "soil"


def getSeason(month):
    if month >= 3 and month <= 5:
        return "Kharif"
    elif month >= 6 and month <= 9:
        return "Rabi"
    elif month >= 10 and month <= 12:
        return "Zaid"
    else:
        return "Kharif"


def getPrice(cropName):
    if cropName.lower() in basePrice.keys():
        return basePrice[cropName.lower()]
    else:
        return 0


@app.route('/', methods=['GET', 'POST'])
def home():
    crop = []
    soil = []
    temp = 0
    humidity = 0
    season = ""

    if request.method == "POST":
        crop = request.form.get("cropName")
        soil = request.form.get("soilType")
        temp = request.form.get("temp")
        humidity = request.form.get("humidity")
        season = request.form.get("season")

    return render_template('index.html', crops=crop, soil=soil, temp=temp, humidity=humidity, season=season)


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    crop = request.form.get('crop')
    soil = request.form.get('soil')
    temp = float(request.form.get('temp'))
    humidity = float(request.form.get('humidity'))

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({'Crop': [crop], 'Soil': [soil], 'Temp': [temp], 'Humidity': [humidity]})

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Render the template with the prediction result
    return render_template('predict.html', crop=crop, prediction=prediction[0])
# Load the trained model and any preprocessing modules if necessary

# Crop Production
model = pickle.load(open(r'crops.pkl', 'rb'))
@app.route('/crop', methods=['GET', 'POST'])
def crop():
    if request.method == 'POST':
        # Handle form submission and processing here
        
        # Retrieve form data
        Nitrogen = request.form.get('Nitrogen')
        Potassium = request.form.get('Potassium')
        Phosphorous = request.form.get('Phosphorous')
        Rainfall = request.form.get('Rainfall')
        PH = request.form.get('PH')
        # Prepare input data as a DataFrame
        
        input_data = pd.DataFrame({'Nitrogen': [Nitrogen], 
                               'Potassium': [Potassium], 
                               'Phosphorous': [Phosphorous], 
                               'Rainfall': [Rainfall], 
                               'PH': [PH], 
                               })
        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        # Render the result template with the prediction results
        return render_template('crop_result.html', prediction=prediction[0])
    else:
        # If it's a GET request, render the crop form template
        return render_template('crop.html')

# to check disease
class DiseaseResult(db.Document):
    crop = db.StringField()
    disease = db.StringField()
    description = db.StringField()
    created_at = db.DateTimeField(default=datetime.now)

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        crop = request.form.get("crop")
        image = request.files["file"]
        # Saving the model
        torch.save(model.state_dict(), 'AIFarming-Deepcoders-main/plant_disease_model.pth')

        # Loading the model
        model = ResNet9(3, 38)
        model.load_state_dict(torch.load('AIFarming-Deepcoders-main/plant_disease_model.pth', map_location=torch.device('cpu')))
        model.eval()

        # Preprocess the image
        image = Image.open(io.BytesIO(image.read()))
        
        # Resize the image to match the input size expected by the model
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),  # Adjust dimensions as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        image = image.unsqueeze(0)

        # Making Prediction
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

        # Fetching Disease Name and Description
        disease_name = disease_dic[class_idx]
        description = disease_dic[class_idx + 38]

        # Store the result in MongoDB
        result = DiseaseResult(crop=crop, disease=disease_name, description=description)
        result.save()

        results = DiseaseResult.objects()

        # Render the template with the retrieved results
        return render_template('result_disease.html', results=results)

    return render_template('disease.html')


# Fertilizer suggestion
@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'POST':
        crop = request.form.get("crop")
        soil = request.form.get("soil")
        N = request.form.get("N")
        P = request.form.get("P")
        K = request.form.get("K")

        # Include other necessary features for prediction
        # For example, if your model expects 7 features, add the remaining 4 features here
        # Additional features could be obtained from the form or computed based on other parameters
        # For demonstration purposes, let's assume the additional features are hardcoded to 0
        additional_features = [0] * 4

        # Combine all features
        features = [N, P, K] + additional_features

        # Convert features to numpy array and reshape
        features = np.array(features).astype(float)
        features = features.reshape(1, -1)

        # Make prediction
        pred = fert.predict(features)

        # Fetching Fertilizer Name
        fertilizer_name = fertilizer_dic[pred[0]]

        return render_template('fertilizer.html', crop=crop, fertilizer=fertilizer_name)

    return render_template('fertilizer.html')

# add Market
@app.route('/market', methods=['GET', 'POST'])
def market():
    if request.method == 'POST':
        fname = request.form.get("fname")
        lname = request.form.get("lname")
        email = request.form.get("email")
        phone = request.form.get("phone")
        address = request.form.get("address")
        croptype = request.form.get("croptype")
        quantity = request.form.get("quantity")
        cropname = request.form.get("cropname")
        msp = request.form.get("msp")
        market = Market(
            fname=fname,
            lname=lname,
            email=email,
            phone=phone,
            address=address,
            croptype=croptype,
            quantity=quantity,
            cropname=cropname,
            msp=msp
        )
        market.save()

        return redirect('/market')

    return render_template('market.html')


# View Market
@app.route('/sell', methods=['GET', 'POST'])
def viewmarket():
    market = Market.objects.all()
    return render_template('sell.html', markets=market)

# forecast
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        # Handle form submission and forecasting logic here
        crop_type = request.form.get('croptype')
        # Retrieve other form inputs as needed
        # For example: soil_type = request.form.get('soiltype')
        # Retrieve other input data based on your form
        
        # Perform forecasting based on the input data
        # For demonstration purposes, let's assume we have a function named 'perform_forecasting'
        # This function takes input data and returns forecasted results
        forecast_data = perform_forecasting(crop_type)  # Pass crop type to the forecasting function
        
        # Render the template with the forecasting results
        return render_template('forecast.html', forecast_data=forecast_data)  # Pass forecasting data to the template
    else:
        # Handle GET request (if needed)
        return render_template('forecast.html')  # Render the empty forecasting form


if __name__ == "__main__":
    app.run(debug=True)
