import streamlit as st
from sklearn import datasets
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit Example")

st.write("""
# Explore different classifier
Which one is the **best**?
         """)

dataset_name = st.sidebar.selectbox("Select Dataset:", ("Iris", "Wine", "Breast Cancer"))

clf_name = st.sidebar.selectbox("Select Algorithm:", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        dataset = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        dataset = datasets.load_breast_cancer()
    else:
        dataset = datasets.load_wine()
    
    X = dataset.data
    y = dataset.target
    return X, y

X, y = get_dataset(dataset_name)

st.write(dataset_name, f"Feature with Shape = {X.shape}")
st.write(pd.DataFrame(X).head())

st.write(dataset_name, f"Target with Shape = {y.shape}, and {pd.Series(y).nunique()} distinct classes")
st.write(pd.Series(y).head())

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        k = st.sidebar.slider("Number of k:", 1, 15)
        params["k"] = k
    elif clf_name == "SVM":
        C = st.sidebar.slider("c value:", 0.01, 10.0)
        params["c"] = C
    else:
        max_depth = st.sidebar.slider("Maximum depth:", 2, 15)
        n_estimators = st.sidebar.slider("Number of estimator:", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimators
    return params

params = add_parameter_ui(clf_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["k"])
    elif clf_name == "SVM":
        clf = SVC(C=params["c"])
    else:
        clf = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimator'],
                                     random_state=42)
    return clf

clf = get_classifier(clf_name, params)

#Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f'Classfier = {clf_name}')
st.write(f'Accuracy = {acc}')

#PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)


