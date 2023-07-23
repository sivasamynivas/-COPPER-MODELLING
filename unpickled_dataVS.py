import pandas as pd
import pickle
import streamlit as st
import numpy as np

def unpickle_dataframe(filename):
    try:
        with open(filename, 'rb') as file:
            df = pickle.load(file)
        print(f"DataFrame successfully unpickled from {filename}.")
        return df
    except Exception as e:
        print(f"Error unpickling DataFrame: {e}")
        return None

# Replace 'path_to_df3.pickle' with the actual file path of your pickled DataFrame.
filename = 'D:\CAPSTONE PROJECTS\CAPS-3 - COPPER MODELLING\df3.pickle'

# Unpickle the DataFrame
df4 = unpickle_dataframe(filename)




# Verify the DataFrame
# if df4 is not None:
#     print("Unpickled DataFrame:")
#     print(df4)


from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Create a StandardScaler object
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df4)

X = df4.drop('status', axis=1)
y = df4['status']

# Print class distribution before oversampling
# print("Class distribution before oversampling:")
# print(Counter(y))

# Instantiate SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Fit and apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution after oversampling
# print("Class distribution after oversampling:")
# print(Counter(y_resampled))

# BUILDING A CLASSIFICATION MACHINE LEARNING MODEL-RANDOM FOREST
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create a Random Forest classifier
# You can customize the parameters as needed (e.g., n_estimators, max_depth, etc.)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the scaled training data
rf_classifier.fit(X_train, y_train)
# # Make predictions on the scaled test data
y_pred = rf_classifier.predict(X_test)
# # Evaluate the model's performance
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# # Print the classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

#TESTING WITH REAL VALUES
XTEST = [5.17,25.00,1,41.00,0.48,1210.00,1668701718,1047.00]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_TEST_scaled = scaler.transform([XTEST])
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict([XTEST])

# print(y_pred)


x1 = df4.drop('selling_price', axis=1)
y1 = df4['selling_price']
# Scale the data to have mean=0 and variance=1
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(x1)
# Split the scaled data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)
# Create a Random Forest Regressor
model1 = RandomForestRegressor(
    n_estimators=100,     # Number of trees in the forest
    criterion='squared_error',  # Mean squared error as the criterion for splitting
    max_depth=None,       # Maximum depth of the trees (None means unlimited)
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    n_jobs=-1,            # Number of CPU cores to use (-1 uses all available cores)
    random_state=42       # Seed for reproducibility
)
# Fit the model to the training data
model1.fit(X1_train, y1_train)
# Make predictions on the test data
y1_pred = model1.predict(X1_test)
from sklearn.metrics import mean_squared_error, r2_score
# Evaluate the model
mse = mean_squared_error(y1_test, y1_pred)
r2 = r2_score(y1_test, y1_pred)

# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R-squared: {r2:.2f}")

#TESTING WITH REAL VALUES
X1TEST = [4.75,30.00,1,4,28.00,0.29,952.00,628377]

scaler = StandardScaler()
X1_train_scaled = scaler.fit(X1_train)
X1_train_scaled = scaler.transform(X1_train)

X1_TEST_scaled = scaler.transform([X1TEST])
model1 = RandomForestRegressor(
    n_estimators=100,     # Number of trees in the forest
    criterion='squared_error',  # Mean squared error as the criterion for splitting
    max_depth=None,       # Maximum depth of the trees (None means unlimited)
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    n_jobs=-1,            # Number of CPU cores to use (-1 uses all available cores)
    random_state=42       # Seed for reproducibility
)
model1.fit(X1_train_scaled, y1_train)

y1_pred = model1.predict(X1_TEST_scaled)

# print(y1_pred)


st.title("INDUSTRIAL COPPER MODELLING")

col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
with col1:
    quantity_tons = st.number_input("Enter the Quantity in Tons")
        
with col2:
    country = st.selectbox("Choose a country",[28.0,25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 89.0, 107.0])
        
with col3:
    item_type = st.selectbox("Choose a item-type \n '5.0 for W', \n '6.0 for WI', \n '3.0 for S', \n '1.0 for Others',\n  '2.0 for PL', \n '0.0 for IPL', \n '4.0 for SLAWR'",[0.0,1.0,2.0,3.0,4.0,5.0,6.0] )
        
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
        

XTEST = [quantity_tons,country,item_type,application,thickness,width,product_ref,selling_price]
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_TEST_scaled = scaler.transform([XTEST])
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict([XTEST])

if st.button("Predict"):
    if y_pred == 0:
        st.markdown("The Predicted Outcome is 0 ")
        st.markdown("THIS IS LOST")

    else:
        st.markdown("The Predicted Outcome is 1")
        st.markdown("THIS IS WON")

            


        

    

            
if options == "Selling Price Prediction":
    col1, col2, col3,col4,col5,col6,col7,col8= st.columns(8)

    with col1:
        quantity_tons = st.number_input("Enter the Quantity in Tons")
    with col2:
        country = st.selectbox("Choose a country",[28.0,25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0, 80.0, 89.0, 107.0])
    with col3:
        status = st.number_input("Enter the Status (0-lost),(1-Won)")
    with col4:
        item_type = st.selectbox("Choose a item-type \n '5.0 for W', \n '6.0 for WI', \n '3.0 for S', \n '1.0 for Others',\n  '2.0 for PL', \n '0.0 for IPL', \n '4.0 for SLAWR'",[0.0,1.0,2.0,3.0,4.0,5.0,6.0] )
    with col5:
        application = st.selectbox("choose a application",[10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0, 29.0, 22.0, 25.0, 40.0, 79.0, 3.0, 99.0, 2.0, 67.0, 5.0, 39.0, 69.0, 70.0, 65.0, 58.0, 68.0] )
    with col6:
        thickness = st.number_input("Enter the Thickness")
    with col7:
        width = st.number_input("Enter the Width")
    with col8:
        product_ref = st.selectbox("Choose Product_ref",[1670798778, 1668701718, 628377, 640665, 611993,1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026, 164336407, 1665572032, 164337175, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1668701725, 1665584320, 1665584642])
    

    newtestdata_reg = [quantity_tons, country, status,item_type, application, thickness, width, product_ref]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train_reg)
    X_train_reg = scaler.transform(X_train_reg)

    # Create a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, n_jobs=-1, random_state=42)
    # Fit the model to the training data
    model.fit(X_train_reg, y_train_reg)

    new_test_data_reg = np.array([newtestdata_reg])
    X_train_reg = scaler.transform(X_train_reg)
    new_test_data_scale = scaler.transform(new_test_data_reg)

    #  Random Forest classifier with best parameters
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  n_jobs=-1,
                                  random_state=42
                                  )
    result_rf = model.fit(X_train_reg, y_train_reg)

    if st.button("Click to Predict"):
        new_predicted_reg = result_rf.predict(new_test_data_scale)
        st.write("Predicted Selling price of Copper : ",new_predicted_reg)

