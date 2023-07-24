import pandas as pd
import pickle
import streamlit as st
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def unpickle_dataframe(filename):
    try:
        with open(filename, 'rb') as file:
            df = pickle.load(file)
        print(f"DataFrame successfully unpickled from {filename}.")
        return df
    except Exception as e:
        print(f"Error unpickling DataFrame: {e}")
        return None


filename = 'D:\CAPSTONE PROJECTS\CAPS-3 - COPPER MODELLING\df3.pickle'


df4 = unpickle_dataframe(filename)
df5 = df4.copy()

# CLASSIFICATION MODEL
X = df4.drop('status', axis=1)
y = df4['status']
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


st.title("INDUSTRIAL COPPER MODELLING")

options = ["ABOUT", "STATUS PREDICTION", "SELLING PRICE PREDICTION", "OUTCOMES"]
selected_option = st.sidebar.selectbox("Select an option:", options)
    
if selected_option == "ABOUT":
    st.title("ABOUT")
    st.write(""" The copper industry deals with less complex data related to sales and pricing.
                    However, this data may suffer from issues such as skewness and noisy data, which
                    can affect the accuracy of manual predictions. Dealing with these challenges manually
                    can be time-consuming & may Not result n optimal pricing decisions. A machine
                    learning regression model can address these issues by utilizing advanced techniques.""")
    
    st.write("""Another area where the copper industry faces challenges is in capturing the leads. A
                lead classification model is a system for evaluating and classifying leads based on
                how likely they are to become a customer .""")
    
    



if selected_option == "STATUS PREDICTION":
    col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
    with col1:
        quantity_tons = st.number_input("Enter the Quantity in Tons")
    with col2:
        country = st.selectbox("Choose a country",[28.0,25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 89.0, 107.0])
    with col3:
        item_type = st.selectbox("Choose a item-type \n '1 for W', \n '4 for WI', \n '2 for S', \n '7 for Others',\n  '3 for PL', \n '5 for IPL', \n '6 for SLAWR'",[0.0,1.0,2.0,3.0,4.0,5.0,6.0] )
    with col4:
        application = st.selectbox("choose a application",[10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0, 29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0, 58.0, 68.0])
    with col5:
        thickness = st.number_input("Enter the Thickness")
    with col6:
         width = st.number_input("Enter the Width")
    with col7:
        product_ref = st.selectbox("Choose Product_ref",[1670798778, 1668701718, 628377, 640665, 611993,1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725, 1665584320, 1665584642])
    with col8:
        selling_price = st.number_input("Enter the Selling Price")

    
    if st.button("Predict"):
    # MODEL BUILDING
        scaler = StandardScaler()
        
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)

        #TESTING THE DATA
        XTEST = [quantity_tons,country,item_type,application,thickness,width,product_ref,selling_price]
        

        X_TEST_scaled = scaler.transform([XTEST])
        y_pred = rf_classifier.predict(X_TEST_scaled )

        
        if y_pred == 0:
            st.markdown("The Predicted Outcome is 0 ")
            st.markdown("THIS IS LOST")
        else:
            st.markdown("The Predicted Outcome is 1")
            st.markdown("THIS IS WON")



# REGRESSION MODEL
if selected_option == "SELLING PRICE PREDICTION":
    col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
    with col1:
        quantity_tons = st.number_input("Enter the Quantity in Tons")
    with col2:
        country = st.selectbox("Choose a country",[28.0,25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 89.0, 107.0])
    with col3:
        status = st.number_input("Enter the Status (0-lost),(1-Won)")
    with col4:
        item_type = st.selectbox("Choose a item-type \n '1 for W', \n '4 for WI', \n '2 for S', \n '7 for Others',\n  '3 for PL', \n '5 for IPL', \n '6 for SLAWR'",[0.0,1.0,2.0,3.0,4.0,5.0,6.0] )
    with col5:
        application = st.selectbox("choose a application",[10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0, 29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0, 58.0, 68.0] )
    with col6:
        thickness = st.number_input("Enter the Thickness")
    with col7:
        width = st.number_input("Enter the Width")
    with col8:
        product_ref = st.selectbox("Choose Product_ref",[1670798778, 1668701718, 628377, 640665, 611993,1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725, 1665584320, 1665584642])

    if st.button("Predict"):
          x1 = df5.drop('selling_price', axis=1)
          y1 = df5['selling_price']

          scaler = StandardScaler()
          X1_scaled = scaler.fit_transform(x1)

          X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)

          model1 = RandomForestRegressor(
                n_estimators=100,    
                criterion='squared_error',  
                max_depth=None,       
                min_samples_split=2,  
                min_samples_leaf=1,   
                n_jobs=-1,            
                random_state=42       
            )

          result = model1.fit(X1_train, y1_train)

          YTEST = [quantity_tons, country, status,item_type, application, thickness, width, product_ref]
          YTEST_scaled = scaler.transform([YTEST])
          y_pred = result.predict(YTEST_scaled )
          st.write("Predicted Selling price of Copper : ",y_pred)

if selected_option == "OUTCOMES":
    st.title("OUTCOMES")
    st.write(""" THIS APPROACH DEALS WITH THE TWO RESULTS 
             \n 1)THE PRICE DETECTION OF A COPPER DEALS WITH A VARIOUS HUGE SPECIFICATIONS ,BY USING A REGRESSION MODEL IT WAS PREDICTED NEUTRALLY IN THIS APPROACH 
             \n 2)BASED ON THE VARIENTS AND LOT OF SPECIFICATIONS IDENTIFYING THE CUSTOMER PREDICTION IS VERY COMPLEX ,BY USING A CLASSIFICATION MODEL IT WAS PREDICTED MORE NEARLY TO THE REAL ONE .""")

