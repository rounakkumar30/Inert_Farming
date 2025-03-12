# 🌾 INERT Farming

## 📌 Problem Statement

World cereal equivalent (CE) food demand is projected to be around **10,094 million tons in 2030** and **14,886 million tons in 2050**, while production is projected to be **10,120 million tons in 2030** and **15,970 million tons in 2050**, having a marginal surplus.

India and China are capturing a large share of global food demand, making efficient farming solutions necessary.

---

## 📊 Implementation Diagram

![Implementation Diagram](photo/flowchart.png)

![AI System](photo/first.png)

---

## 🌱 Crop Yield Prediction

The model predicts crop yield based on the following parameters:

- **Nitrogen**
- **Phosphorous**
- **State**
- **City**
- **pH**
- **Rainfall**

📡 *State and City parameters are used to fetch weather data (temperature, rainfall, humidity) via an API call.*

The **Random Forest** algorithm is used to provide the most accurate yield predictions. The results are displayed on the screen.

![Crop Yield Prediction](https://user-images.githubusercontent.com/75557390/177081039-dca86c74-da61-4364-b01a-c257f0d219ed.png)

---

## 🌾 Fertilizer Prediction

A pre-existing fertilizer dataset is used for training and testing.

### 🔹 User Inputs:
- **Nitrogen**
- **Phosphorous**
- **Soil Type**

🌦️ *Weather details (Temperature, Humidity, Moisture) are fetched via API calls.*

### 🔹 Output:
- Recommended **conventional fertilizer**
- Suggested **organic alternatives**
- **General information** about fertilizers
- **Dosage recommendations** for crops

---

## 💰 Price Prediction

Each year, the Government releases the **Wholesale Price Index (WPI)**, which is used to estimate crop prices.

📈 **Formula:**  
`Current Price = WPI * Base Price (for the ongoing year)`

Using historical data, the model predicts:

- **Maximum price**
- **Average predicted price**
- **Minimum price**

A **graph** is plotted showing:
- **Projected crop prices** (left)
- **Historical price trends (2012-2019)** (right)

🧠 **Model Used:** **SARIMAX** (Time series forecasting)  
✅ Can predict prices **4-5 years into the future** with high accuracy.

---

## 🌿 Crop Disease Prediction

👨‍🌾 *Farmers can upload an image of a diseased crop on the website.*  

🔬 **Deep Learning models** analyze the image and detect the disease, providing:
- **Disease details**
- **Possible causes**
- **Recommended cure**

---

## 📰 Farmer News Portal

Farmers can stay updated with **daily agricultural news** via a web portal.

- **News is dynamically fetched** from various sources using **web crawling**.
- Clicking on **"Read Full Article"** redirects users to the original source.

---

## 🏪 Marketplace

Farmers can **sell crops directly to buyers**, reducing middlemen costs.

📌 **User Inputs:**
- **Crop name**
- **Asking price**
- **Quantity available**
- **Contact details**

Buyers can connect with farmers, ensuring better profits and direct sales.

---

## ⚙️ Project Setup & Execution

### 🔹 Steps to Run the Project:
1. **Install dependencies**:  
   ```sh
   pip install -r requirements.txt
   ```

2. **Run all Jupyter notebooks** (inside the `notebook` folder) to generate model files.

3. **Verify required pickle files** after running the notebooks.

4. **Run the application**:  
   ```sh
   python app.py
   ```

---

## 📚 References

- **Crop yield prediction using machine learning algorithms** - *International Journal of Recent Technology and Engineering (IJRTE)*
- **Crop Condition Assessment using Machine Learning** - *International Journal of Recent Technology and Engineering (IJRTE)*
- **Open Government Data (OGD) Platform India**
- **Kaggle: Your Machine Learning and Data Science Community**

---

🚀 **Developed by [Rounak Kumar](https://github.com/rounakkumar30)**  

## 📜 License

This project is licensed under the **MIT License**.

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

---
📌 *For any queries, feel free to raise an issue or contribute to the project!*
