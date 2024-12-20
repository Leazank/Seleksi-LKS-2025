import streamlit as st
from building_model import KNN
from sklearn import datasets
import pandas as pd
import numpy as np

cancer = datasets.load_iris()
df = pd.concat([pd.DataFrame(cancer.data, columns=cancer.feature_names), pd.Series(cancer.target, name='target')], axis=1)
column_names = [x for x in datasets.load_iris().feature_names]
column_names.append('target')

# Author text
st.sidebar.markdown('<h5 style="color: white;"> Author : Shabri Siregar </h5>', unsafe_allow_html=True)

# Sidebar for user input selection
st.sidebar.markdown('<h1 style="color: white;">Select One output and at least one input Variable</h1>', unsafe_allow_html=True)
# Select output variable
output_variable_model = st.sidebar.selectbox('Select One output Variable', column_names)

# Select input variables to predict the target variable (output)
input_variables_model = st.sidebar.multiselect('Select at least one input Variable', column_names)

# if user select all
is_select_all = st.sidebar.checkbox('Select all columns to input (except target variable)')

if is_select_all:
    column_names_no_target = column_names[:-1]
    input_variables_model = column_names_no_target

if not output_variable_model or not input_variables_model:
    st.warning('Select One output and at least one input Variable to start.')

# User option for setting the rate of test data
training_ratio = st.sidebar.text_input('Input training ratio (from 0 - 1)', 0.7)

if (not training_ratio) or (float(training_ratio) <= 0) or (float(training_ratio) >= 1):
    st.sidebar.error('Error! Make sure your training ratio is not empty and stand between 0 to 1')
number_of_k = st.sidebar.slider('Select the number of k', 1, 20, 1, 1)

#Training model process
df_shuffle = df.sample(frac=1, random_state=42)
training_dataset = df_shuffle.iloc[:int(float(training_ratio)*len(df_shuffle))]
testing_dataset = df_shuffle.iloc[int(float(training_ratio)*len(df_shuffle)):]

st.subheader('Training Dataset')
st.write(training_dataset[input_variables_model])
st.write(f'{training_dataset[input_variables_model].shape[0]} rows and {training_dataset[input_variables_model].shape[1]} columns')


st.subheader('Testing Dataset')
st.write(testing_dataset[input_variables_model])
st.write(f'{testing_dataset[input_variables_model].shape[0]} rows and {testing_dataset[input_variables_model].shape[1]} columns')

used_col = list(pd.concat([pd.Series(input_variables_model), pd.Series(output_variable_model)]).reset_index(drop=True))

clf = KNN(k=number_of_k)
clf.fit(training_dataset[used_col], label=str(output_variable_model))
y_pred = clf.multiple_predict(testing_dataset[used_col].drop('target', axis=1))
st.write('Model Accuracy: ', clf.accuracy(testing_dataset[used_col]['target'], y_pred))


