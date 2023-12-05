import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons.csv')
# printing the first 5 rows of the dataframe
parkinsons_data.head()
# number of rows and columns in the dataframe
parkinsons_data.shape
# getting more information about the dataset
parkinsons_data.info()
# checking for missing values in each column
parkinsons_data.isnull().sum()
# getting some statistical measures about the data
parkinsons_data.describe()
# distribution of target Variable
parkinsons_data['status'].value_counts()
# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status'
print(Y)
print(X)X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = svm.SVC(kernel='linear')
# training the SVM model with training data
model.fit(X_train, Y_train)
# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
input_data =
(197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,
0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-
7.348300,0.177551,1.743867,0.085569)
# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
 print("The Person does not have Parkinsons Disease")
else:
 print("The Person has Parkinsons")
import pickle
filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
CODE FOR WEBPAGE
import streamlit as st
from streamlit_option_menu import option_menu
# set page config
st.set_page_config(page_title="Parkinson's Disease Prediction App",
page_icon=":clipboard:", layout="wide", initial_sidebar_state="expanded",
menu_items={"Get Help": "https://www.streamlit.io/"}, )
# rest of the code
import pickle
import pandas as pd
#loading the saved model
parkinsons_model = pickle.load(open('D:/desktop/Parkinsons disease prediction/saved
model/parkinsons_model.sav','rb'))
#sidebar for navigate
with st.sidebar:
 selected = option_menu('Minor Project -IV',
 ['Home','Parkinsons Disease Prediction'],
 icons = ['house','person'],
 default_index= 0)
# Home Page
if selected == 'Home':
 st.title('Welcome to Parkinsons Disease Prediction App')
 st.write('This app predicts whether a person has Parkinsons Disease or not based on
their health attributes')
# Parkinson's Prediction Page
elif selected == 'Parkinsons Disease Prediction':
 # page title
 st.title("Parkinson's Disease Prediction using ML")
 # file upload
 st.header('Upload CSV file')
 file = st.file_uploader('Choose a CSV file', type='csv')
 if file is not None:
 df = pd.read_csv(file)
 # show uploaded data
 st.write('*Data uploaded:*')
 st.write(df)
 # select row
 st.header('Select a row to predict result')
 row_index = st.selectbox('Row', df.index)
 # code for prediction
 st.header('Prediction Result')
 if st.button("Predict"):
 parkinsons_prediction = parkinsons_model.predict([[
 df.at[row_index, 'MDVP:Fo(Hz)'],
 df.at[row_index, 'MDVP:Fhi(Hz)'],
 df.at[row_index, 'MDVP:Flo(Hz)'],
 df.at[row_index, 'MDVP:Jitter(%)'],
 df.at[row_index, 'MDVP:Jitter(Abs)'],
 df.at[row_index, 'MDVP:RAP'],
 df.at[row_index, 'MDVP:PPQ'],
 df.at[row_index, 'Jitter:DDP'],
 df.at[row_index, 'MDVP:Shimmer'],
 df.at[row_index, 'MDVP:Shimmer(dB)'],
 df.at[row_index, 'Shimmer:APQ3'],
 df.at[row_index, 'Shimmer:APQ5'],
 df.at[row_index, 'MDVP:APQ'],
 df.at[row_index, 'Shimmer:DDA'],
 df.at[row_index, 'NHR'],
 df.at[row_index, 'HNR'],
 df.at[row_index, 'RPDE'],
 df.at[row_index, 'DFA'],
 df.at[row_index, 'spread1'],
 df.at[row_index, 'spread2'],
 df.at[row_index, 'D2'],
 df.at[row_index, 'PPE']]])
 # show predicted result
 if parkinsons_prediction[0] == 1:
 st.write(f"The person in row {row_index + 1} has Parkinson's disease",
font_size=20)
 else:
 st.write(f"The person in row {row_index + 1} does not have Parkinson's
disease", font_size=20)
 else:
 st.write('Please upload a CSV file to see prediction results')
