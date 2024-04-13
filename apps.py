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


# importing and Loading Pickle File
model = pickle.load(open(r'crops.pkl', 'rb'))
fert = pickle.load(open(r'AIFarming-Deepcoders-main\fertilizer.pkl', 'rb'))



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


# to check disease
@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        crop = request.form.get("crop")
        image = request.files["image"]
        model = ResNet9(3, 38)
        model.load_state_dict(torch.load(
            'AIFarming-Deepcoders-main\\plant_disease_model.pth', map_location=torch.device('cpu')))
        model.eval()

        # Preprocessing the image
        image = Image.open(io.BytesIO(image.read()))
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        image = image.unsqueeze(0)

        # Making Prediction
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

        # Fetching Disease Name
        disease_name = disease_dic[class_idx]

        # Fetching Description
        description = disease_dic[class_idx + 38]

        # Pass data to the template
        return render_template('disease.html', crop=crop, disease=disease_name, description=description)

    return render_template('disease.html')

if __name__ == "__main__":
    app.run(debug=True)
