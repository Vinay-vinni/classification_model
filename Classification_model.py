###### Libraries #######

# Base Libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########

data = pd.read_csv('Data.csv')

########### Loading Trained Model Files ########
model = joblib.load("knn_Add_Click")
model_ohe = joblib.load("data_ohe")
model_sc = joblib.load("data_sc")
model_pca = joblib.load("data_pca")
########## UI Code #############

# Ref: https://docs.streamlit.io/library/api-reference

# Title
st.header("Identifying weather the person clicked the add or not:")

# Image
with st.columns(3)[1]:
    st.image("https://dgbijzg00pxv8.cloudfront.net/ab1fe4b5-510e-4405-ad12-234afb1dfda9/000000-0000000000/39072274470874426121284330130872841955733143294728805461369592993420210902166/ITEM_PREVIEW1.png", width=400)

# Description
st.write("""Built a Predictive model in Machine Learning to Identify weather the User clicked the add or not.
         Sample Data taken as below shown.
""")

# column name spaces repalcing with ( _ )

data.columns = data.columns.str.replace(" ","_")


# timestamp is divided

from datetime import datetime

data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d-%m-%Y %H.%M")

data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Weekday'] = data['Timestamp'].dt.dayofweek
data['Hour'] = data['Timestamp'].dt.hour
data = data.drop(['Timestamp'], axis=1)



# deleting output col

del data['Clicked_on_Ad']






# Data Display
st.dataframe(data.head())
st.write("From the above data , Add clicked is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get weather the user clicked or not:")

col1,  col2,  col3 = st.columns(3) # value inside brace defines the number of splits
col4,  col5,  col6 = st.columns(3)
col7,  col8,  col9 = st.columns(3)
col10, col11, col12 = st.columns(3)


with col1:
    Daily_Time_Spent_on_Site = st.number_input("Enter Daily_Time_Spent_on_Site :",min_value=0)
    st.write(Daily_Time_Spent_on_Site)

with col2:
    print("If Age between 19 and 35 give Adults")
    print("If Age between 36 and 50 give Middle_Aged")
    print("If Age greater than 50 give Senior_Citizen")
    Age= st.number_input("Enter Age:",min_value=0)
    st.write(Age)

with col3:
     Area_Income = st.number_input("enter Area_Income:",min_value=0)
     st.write(Area_Income)

with col4:
     Daily_Internet_Usage= st.number_input("enter Daily_Internet_Usage:")
     st.write(Daily_Internet_Usage)



with col5:
     Ad_Topic_Line= st.selectbox("enter Ad_Topic_Line :",data.Ad_Topic_Line.unique())
     st.write(Ad_Topic_Line)

with col6:
     City= st.selectbox("enter City:",data.City.unique())
     st.write(City)

with col7:
     Gender = st.selectbox("Enter Gender:",data.Gender.unique())
     st.write(Gender)

with col8:
     Country = st.selectbox("Enter Country:",data.Country.unique())
     st.write(Country)

with col9:
     Month= st.number_input("enter Month:",min_value=0)
     st.write(Month)

with col10:
     Day= st.number_input("enter Day:",min_value=0)
     st.write(Day)

with col11:
     Weekday= st.number_input("enter Weekday:",min_value=0)
     st.write(Weekday)

with col12:
     Hour=st.number_input("enter Hour:",min_value=0)
     st.write(Hour)
    





    

###### Predictions #########

if st.button("Check here"):
    st.write("Data Given:")
    values = [Daily_Time_Spent_on_Site, Age,Area_Income,Daily_Internet_Usage,Ad_Topic_Line,City,Gender,Country,Month,Day,Weekday,Hour]
    record =  pd.DataFrame([values],
                           columns = ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income',
                                     'Daily_Internet_Usage', 'Ad_Topic_Line', 'City', 'Gender', 'Country',
                                     'Month', 'Day', 'Weekday', 'Hour'])
    

     # Preprocess input data

    for i in range(len(record)):
     val = record.Age
     if val[i]>=19 and val[i]<=35:
         record.Age[i] = 'Adults'
     elif val[i]>35 and val[i]<=50:
          record.Age[i] = 'Middle_Aged'
     elif val[i]>50:
          record.Age[i] = 'Senior_Citizen'

     record.Age.replace({'Adults':0,'Middle_Aged':1,'Senior_Citizen':2}, inplace=True)

     for i in range(len(record)):
          if record.Hour[i] >= 0  and record.Hour[i] <=5:
               record['Hour'][i] ='mid_night'
          elif record.Hour[i] >= 6 and record.Hour[i] <=  11:
               record['Hour'][i] = 'morning'
          elif record.Hour[i] >= 12 and record.Hour[i] <= 16:
               record['Hour'][i] = 'afternoon'
          elif record.Hour[i] >= 17 and record.Hour[i] <= 19:
               record['Hour'][i] = 'evening'
          else :
               record['Hour'][i] = 'night'

     for col in record.columns:
          if record[col].dtype =='object':
               record[col] = record[col].str.lower()

     record.Hour.replace({'morning':0,'afternoon':1,'evening':2,'night':3,'mid_night':4}, inplace=True)
     record.Gender.replace({'male':0,'female':1}, inplace= True)

     ohedata = model_ohe.transform(record[['Ad_Topic_Line','City','Country']]).toarray()
     ohedata = pd.DataFrame(ohedata, columns=model_ohe.get_feature_names_out())
     record = pd.concat([record.iloc[:,0:], ohedata], axis=1)
     record.drop(['Ad_Topic_Line','City','Country'], axis = 1,inplace =True)

     st.dataframe(record)


     record.iloc[:, [0,2,3,5,6,7,8]] = model_sc.transform(record.iloc[:, [0,2,3,5,6,7,8]])

     data_pca = model_pca.transform(record)
     data_pca = pd.DataFrame(data_pca[:,:43])
     data_pca_array = data_pca.values

     record_values =record.values
     clicked = model.predict(data_pca_array)
     clicked_str = str(clicked[0])

     st.subheader("Add clicked or not :")
    if clicked_str==0:
        print("Ad Not Clicked")
    else:
        print("Ad Clicked")


 


