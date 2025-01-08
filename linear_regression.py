import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Linear Regression on CSV Data with Feature Selection and Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display data preview before NaN handling
    st.write("Data Preview (Before NaN Removal)", df.head())

    # Remove rows with NaN values
    df = df.dropna()
    st.write("Data Preview (After NaN Removal)", df.head())

    # Filter numeric columns only
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        st.write("No numeric columns found in the dataset. Unable to perform correlation-based feature selection.")
    else:
        # Select target column
        columns = df.columns.tolist()
        target = st.selectbox("Select target column", columns)

        # Ensure target is a numeric column
        if target not in numeric_columns:
            st.write(f"Target column {target} is not numeric. Please select a numeric column.")
        else:
            # Automatically select features based on correlation with target
            correlation = df[numeric_columns].corr()[target].sort_values(ascending=False)
            # Remove the target itself from features
            features = correlation.index[1:].tolist()  # Remove target from features
            st.write("Automatically selected feature columns based on correlation:", features)

            # Allow the user to edit or select features
            st.write("### Edit or Select Features")
            selected_features = st.multiselect("Select features", features, default=features)

            if selected_features:
                # Prepare data
                X = df[selected_features]
                y = df[target]

                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Display coefficients
                st.write("Coefficients", model.coef_)
                st.write("Intercept", model.intercept_)

                # Prediction input form
                st.write("### Predict Target Value")
                input_data = {}
                for feature in selected_features:
                    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                if st.button("Predict"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    st.write(f"Predicted {target}: {prediction}")

                # Plot results
                if len(selected_features) == 1:  # Simple linear regression
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=X.iloc[:, 0], y=y)
                    plt.plot(X, model.predict(X), color='red')
                    plt.xlabel(selected_features[0])
                    plt.ylabel(target)
                    plt.title("Linear Regression Fit")
                    st.pyplot(plt)
                else:
                    st.write("Plotting is only available for simple linear regression.")

else:
    st.write("Awaiting CSV file upload.")

# Run the Streamlit app with: streamlit run app.py
