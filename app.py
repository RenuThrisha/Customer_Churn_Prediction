import tensorflow
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model


##lloading the trained mode1
model=load_model('model.h5')



## load the encoders and scalar
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


with open ('onehot_encoder_geography.pkl','rb') as file:
    ohe_geo=pickle.load(file)


with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Customer churn prediction')

geography=st.selectbox('Geography',ohe_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credict Score')
estimated_salary=st.number_input('est sal')
tenure=st.slider('Tenture',0,10)
num_of_products=st.slider('No of prodcutcs',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is active number',[0,1])

# Prepare the input data
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})



geo_encoded=ohe_geo.transform(input_df[['Geography']])
geo_df=pd.DataFrame(geo_encoded.toarray(),columns=ohe_geo.get_feature_names_out(['Geography']))

input_df=pd.concat([input_df.drop('Geography',axis=1),geo_df],axis=1)

scaled_input=scaler.transform(input_df)

prediction=model.predict(scaled_input)

prediction_pro=prediction[0][0]

if prediction_pro>0.5:
    st.write('churn.')
else:
    st.write('not likely to churn')




