# Customer Churn Prediction using ANN

A machine learning project that predicts whether a customer is likely to **churn** using an **Artificial Neural Network (ANN)**. The project includes data preprocessing, feature encoding, feature scaling, model training, and a **Streamlit web application** for making predictions.

## Features

- Customer churn prediction using ANN
- Gender encoding using Label Encoder
- Geography encoding using One-Hot Encoder
- Feature scaling using StandardScaler
- Trained TensorFlow/Keras model
- Interactive Streamlit interface
- Real-time churn prediction based on customer details

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Pickle

## Input Features

The model uses the following customer information:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## Project Structure

```text
sec13_ANN/
│
├── app.py
├── model.h5
├── Churn_Modelling.csv
├── label_encoder_gender.pkl
├── onehot_encoder_geography.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

## How It Works

```text
Customer Details
       ↓
Data Preprocessing
       ↓
Gender Encoding
       ↓
Geography One-Hot Encoding
       ↓
Feature Scaling
       ↓
ANN Model
       ↓
Churn Probability
       ↓
Churn / Not Likely to Churn
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd sec13_ANN
```

Create and activate a virtual environment:

```bash
python -m venv annenv
```

For Windows:

```bash
annenv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Enter the customer's details and the model will predict whether the customer is likely to churn.

## Model Output

The model produces a churn probability.

- Probability > 0.5 → **Churn**
- Probability ≤ 0.5 → **Not likely to churn**

## Dataset

The project uses the **Churn Modelling** dataset containing customer information and a target variable indicating whether the customer exited the bank.

## Future Improvements

- Improve model accuracy through hyperparameter tuning
- Add model performance metrics and visualizations
- Deploy the application online
- Add probability visualization to the Streamlit interface

## License

This project is intended for educational and learning purposes.
