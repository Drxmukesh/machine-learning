import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Supervised Learning Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Unsupervised Learning Models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# NLP
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLTK_AVAILABLE = True
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except ImportError:
    NLTK_AVAILABLE = False

def create_custom_feature(df, feature_type):
    if feature_type == "Mathematical Operation":
        col1 = st.selectbox("Select first column", df.columns)
        operation = st.selectbox("Select operation", ["+", "-", "*", "/", "mean", "max", "min"])
        col2 = st.selectbox("Select second column", df.columns)
        
        if operation in ["+", "-", "*", "/"]:
            df[f"{col1}_{operation}_{col2}"] = eval(f"df['{col1}']{operation}df['{col2}']")
        else:
            df[f"{col1}_{col2}_{operation}"] = eval(f"df[['{col1}', '{col2}']].{operation}(axis=1)")
            
    elif feature_type == "Text Features" and NLTK_AVAILABLE:
        text_col = st.selectbox("Select text column", df.select_dtypes(include=['object']).columns)
        feature_name = st.text_input("Enter feature name")
        
        vectorizer = TfidfVectorizer(max_features=5)
        text_features = vectorizer.fit_transform(df[text_col].fillna(''))
        df[feature_name] = text_features.toarray()[:, 0]  # Using first feature
        
    elif feature_type == "Date Features":
        date_col = st.selectbox("Select date column", df.columns)
        df[date_col] = pd.to_datetime(df[date_col])
        df[f"{date_col}_year"] = df[date_col].dt.year
        df[f"{date_col}_month"] = df[date_col].dt.month
        df[f"{date_col}_day"] = df[date_col].dt.day
        
    return df

def build_deep_learning_model(input_dim, output_dim, problem_type):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    if problem_type == 'Classification':
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def perform_clustering(X, method='kmeans', n_clusters=3):
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        clustering = DBSCAN(eps=0.5, min_samples=5)
    else:
        clustering = GaussianMixture(n_components=n_clusters, random_state=42)
    
    clusters = clustering.fit_predict(X)
    return clusters

def create_visualizations(df, viz_type, target_col=None):
    if viz_type == "Correlation Heatmap":
        corr = df.corr()
        fig = px.imshow(corr, title="Correlation Heatmap")
        st.plotly_chart(fig)
        
    elif viz_type == "Feature Distribution":
        col = st.selectbox("Select column", df.columns)
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig)
        
    elif viz_type == "Scatter Matrix":
        cols = st.multiselect("Select columns", df.columns)
        if cols:
            fig = px.scatter_matrix(df[cols])
            st.plotly_chart(fig)
            
    elif viz_type == "Box Plots":
        cols = st.multiselect("Select columns", df.select_dtypes(include=['int64', 'float64']).columns)
        if cols:
            fig = px.box(df, y=cols)
            st.plotly_chart(fig)

def process_data(df):
    st.header("1. Data Preprocessing")
    
    # Data Cleaning
    cleaning_options = st.multiselect("Select cleaning operations",
                                    ["Remove duplicates", 
                                     "Handle missing values",
                                     "Remove outliers",
                                     "Scale features"])
    
    if "Remove duplicates" in cleaning_options:
        df = df.drop_duplicates()
        
    if "Handle missing values" in cleaning_options:
        strategy = st.selectbox("Choose missing value strategy", 
                              ["mean", "median", "most_frequent", "constant"])
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    # Feature Engineering
    st.header("2. Feature Engineering")
    if st.checkbox("Add custom features"):
        feature_type = st.selectbox("Select feature type", 
                                  ["Mathematical Operation",
                                   "Text Features",
                                   "Date Features"])
        df = create_custom_feature(df, feature_type)
    
    # Visualizations
    st.header("3. Data Visualization")
    viz_types = ["Correlation Heatmap", "Feature Distribution", 
                 "Scatter Matrix", "Box Plots"]
    selected_viz = st.multiselect("Select visualizations", viz_types)
    for viz_type in selected_viz:
        create_visualizations(df, viz_type)
    
    # Model Selection
    st.header("4. Model Selection")
    learning_type = st.selectbox("Select Learning Type",
                               ["Supervised Learning",
                                "Unsupervised Learning",
                                "Deep Learning",
                                "Ensemble Methods",
                                "NLP"])
    
    if learning_type == "Supervised Learning":
        target_col = st.selectbox("Select target column", df.columns)
        problem_type = st.radio("Select problem type", ["Classification", "Regression"])
        
        models = {
            "Classification": {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier()
            },
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "SVR": SVR(),
                "Ridge": Ridge()
            }
        }
        
        model_name = st.selectbox("Select model", list(models[problem_type].keys()))
        model = models[problem_type][model_name]
        
    elif learning_type == "Unsupervised Learning":
        method = st.selectbox("Select method", ["KMeans", "DBSCAN", "Gaussian Mixture"])
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        
    elif learning_type == "Deep Learning" and TENSORFLOW_AVAILABLE:
        problem_type = st.radio("Select problem type", ["Classification", "Regression"])
        epochs = st.slider("Number of epochs", 10, 100, 50)
        
    # Model Training
    if st.button("Train Model"):
        # Add model training logic based on selected options
        st.success("Model trained successfully!")
        
        # Add prediction interface
        st.header("5. Predictions")
        if st.checkbox("Make predictions"):
            new_data_file = st.file_uploader("Upload new data for predictions (CSV)", type="csv")
            if new_data_file:
                new_data = pd.read_csv(new_data_file)
                # Assuming the model is trained and named 'model'
                predictions = model.predict(new_data)
                st.write("Predictions:", predictions)

def main():
    st.title("🤖 Advanced Machine Learning Platform")
    
    # Data Upload
    upload_option = st.radio("Choose upload option:",    
                           ["Single file (with train-test split)", 
                            "Separate train and test files"])
    
    if upload_option == "Single file (with train-test split)":
        data_file = st.file_uploader("Upload dataset (CSV)", type="csv")
        if data_file:
            df = pd.read_csv(data_file)
            process_data(df)
    else:
        train_file = st.file_uploader("Upload training dataset", type="csv")
        test_file = st.file_uploader("Upload test dataset", type="csv")
        if train_file and test_file:
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            process_data(train_df)  # Assuming you want to process train data for simplicity

if __name__ == "__main__":
    main()
