
import streamlit as st
import pandas as pd
import requests

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

# Page configuration
st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Prediction")
st.write(
    "Predict product sales using a trained "
    "Random Forest regression model."
)

st.subheader("Upload Product and Store Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing product and store details",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded CSV file
    input_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(input_data)

    # Prediction button
    if st.button("Predict Sales"):

        try:
            # Convert data into JSON records
            payload = input_data.to_dict(
                orient="records"
            )

            # Send request to Flask API
            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:

                predictions = response.json()["predictions"]

                # Display predictions
                results = input_data.copy()
                results["Predicted_Sales"] = predictions

                st.success(
                    "Sales predictions generated successfully!"
                )

                st.subheader("Prediction Results")
                st.dataframe(results)

                # Download prediction results
                csv = results.to_csv(index=False)

                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="superkart_predictions.csv",
                    mime="text/csv"
                )

            else:
                st.error(
                    f"Prediction failed: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            st.error(
                f"Unable to connect to Flask API: {e}"
            )
        except (ValueError, KeyError) as e:
            st.error(
                f"Invalid prediction response: {e}"
            )
