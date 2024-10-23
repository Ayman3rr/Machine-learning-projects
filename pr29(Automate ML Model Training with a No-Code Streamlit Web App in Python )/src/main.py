import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR  # SVR for regression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from ml_utility import (read_data, preprocess_data, train_model, evaluate_model)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Automate ML",
    page_icon="🧠",
    layout="centered"
)

st.title("🤖 No Code ML Model Training")

# Option for user to upload a CSV or Excel file
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Define function to read uploaded files
def load_data(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith('xlsx'):
        return pd.read_excel(file)
    else:
        return None

# Read the dataset if the user uploads a file
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.dataframe(df.head())  # Display first few rows of the dataframe

        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax"]

        model_dictionary = {# تعريف نماذج التصنيف
    "Logistic Regression": LogisticRegression(solver='liblinear', penalty='l1'),
    "Support Vector Classifier": SVC(kernel='sigmoid', gamma=1.0),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=2),
    "AdaBoost Classifier": AdaBoostClassifier(n_estimators=50, random_state=2),
    "Bagging Classifier": BaggingClassifier(n_estimators=50, random_state=2),
    "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=50, random_state=2),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=50, random_state=2),
    "XGBoost Classifier": XGBClassifier(n_estimators=50, random_state=2),
# تعريف نماذج الانحدار
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Support Vector Regression": SVR(kernel='linear', C=1.0),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5),
    "K Neighbors Regressor": KNeighborsRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=2),
    "AdaBoost Regressor": AdaBoostRegressor(n_estimators=50, random_state=2),
    "Bagging Regressor": BaggingRegressor(n_estimators=50, random_state=2),
    "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=50, random_state=2),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=50, random_state=2),
    "XGBoost Regressor": XGBRegressor(n_estimators=50, random_state=2)
            
        }

        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name")

        # Train the model if button is clicked
        if st.button("Train the Model"):
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
            model_to_be_trained = model_dictionary[selected_model]
            model = train_model(X_train, y_train, model_to_be_trained, model_name)
            accuracy = evaluate_model(model, X_test, y_test)
            st.success("Test Accuracy: " + str(accuracy))

else:
    st.info("Please upload a CSV or Excel file to get started.")
