import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model import KNN, Cross_Validation
from model import train_test_split, stratified_split

data = pd.read_csv('cleaned.csv').drop('Unnamed: 0', axis=1)

html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;">Tools Buat Mas Adit si Botanis </h2>
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
st.header("Data Overview")
st.write(data.head())

st.sidebar.title("== **Customize model** ==")
st.sidebar.header("Customize K-Nearest Neighbor")
training_ratio = st.sidebar.number_input("Training Ratio")
if training_ratio <= 0.00:
    st.error("Training Ratio nya yang bener woi!!")
num_k = st.sidebar.slider("Select the number of k", 1, 20, 3)

st.sidebar.header("Customize Cross Validation")
stay_default = st.sidebar.checkbox("Stay default")
st.sidebar.write("------------------")
num_folds = st.sidebar.number_input("Number of folds")
rand = st.sidebar.checkbox("Use Randomize")
strat = st.sidebar.checkbox("Use Stratification")
n_range = st.sidebar.number_input("Range of k to test")

if stay_default:
    num_folds = 5
    rand = True
    strat = True

X_train, X_test, y_train, y_test = stratified_split(data, data.drop('species', axis=1).columns, target='species', test_ratio=training_ratio)

st.write(f'Training dataset shape: {X_train.shape}')
st.write(f'Testing dataset shape: {X_test.shape}')

knn = KNN(k=num_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
st.write(f'Model performance: {knn.accuracy(y_test, y_pred)}')

if st.sidebar.button("Run Cross Validation"):
    acc_k = []
    for i in range(1, int(n_range)):
        knn_cv = KNN(k=i)
        cv = Cross_Validation(knn_cv, int(num_folds), rand, strat)
        cv.fit(data, 'species')
        acc_k.append((cv.run_cv(), i))
    st.sidebar.write(f'Best k: {sorted(acc_k)[-1][1]}')


st.header("Predict new iris")
petal_length = st.number_input("Petal length", 0)
petal_width = st.number_input("Petal width", 0)
sepal_length = st.number_input("Sepal length", 0)
sepal_width = st.number_input("Sepal width", 0)

arr_test = pd.Series([petal_length, petal_width, sepal_length, sepal_width])

if 0 in arr_test.values:
    st.warning("We itu ada yang 0 nilainya!")

if st.button("Predict"):
    st.success(f'Iris type is {knn.single_predict(arr_test)}')






