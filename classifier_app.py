import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

#-new
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# st.set_option('deprecation.showPyplotGlobalUse', False)
st.write('fixing locally')

def split(df, target_column):
    y = df[[target_column]]
    x = df.drop(columns=[target_column])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def plot_metrics(model, metrics_list, y_test, y_pred):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(10, 7))
        disp.plot(ax=ax)
        st.pyplot(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        disp = RocCurveDisplay.from_predictions(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        disp.plot(ax=ax)
        st.pyplot(fig)
    
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        disp.plot(ax=ax)
        st.pyplot(fig)

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")

    st.sidebar.subheader("Choose Dataset")
    dataset = st.sidebar.selectbox("dataset", ("mushrooms", "Iris"))

    df = pd.read_csv("./{}.csv".format(str(dataset)))
    target_column  = st.sidebar.selectbox("target column", df.columns)
    
    # class_names = list(df[[target_column]].unique())

    st.write(df[[target_column]])

    labelencoder=LabelEncoder()
    for col in df.columns:
        df[col] = labelencoder.fit_transform(df[col])

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

    
    x_train, x_test, y_train, y_test = split(df, target_column)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("dataset", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "K-nearest neighbors", "Gaussian Naive Bayes"))

    
    if classifier == 'Gaussian Naive Bayes':
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Gaussian Naive Bayes")
            model = GaussianNB()
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(model, metrics, y_test, y_pred)

    if classifier == 'K-nearest neighbors':
        n_neighbors = st.sidebar.slider("number of neighbors", 12, 30, key='n_neigbbours')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("K-nearest neighbors")
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(model, metrics, y_test, y_pred)


    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(model, metrics, y_test, y_pred)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(model, metrics, y_test, y_pred)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(model, metrics, y_test, y_pred)

   
       

if __name__ == '__main__':
    main()


