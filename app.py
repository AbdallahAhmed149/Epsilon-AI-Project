import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
from streamlit_lottie import st_lottie

# Page Configuration & CSS
st.set_page_config(page_title="Kemet Flight Predictor", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background-color: #004aad; 
        color: white; 
        font-weight: bold; 
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #003080;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper Functions
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


@st.cache_data
def load_data():
    df = pd.read_csv("data/airlines_flights_data.csv")
    if "index" in df.columns:
        df.drop("index", axis=1, inplace=True)
    return df


@st.cache_resource
def load_models():
    # Cache the models so they don't reload on every button click
    model = joblib.load("models/XGBoost.pkl")
    encoder = joblib.load("models/onehot_encoder.pkl")
    scaler = joblib.load("models/standard_scaler.pkl")
    return model, encoder, scaler


# Load Data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found! Please ensure 'data/airlines_flights_data.csv' is in the correct directory.")
    st.stop()

# ==========================================
# 3. Sidebar Navigation
# ==========================================
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio("Go to", ["🔮 Price Predictor", "📊 Data Dashboard", "👨‍💻 About Developer"])

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit, XGBoost, and Plotly.")

# Page 1: Price predictor (Main Page)
if page == "🔮 Price Predictor":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🔮 Flight Ticket Price Predictor")
        st.markdown("Enter your flight details below. Our XGBoost model will analyze market trends to predict your estimated ticket price.")

    with col2:
        # Fun animation
        lottie_airplane = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jzztebiu.json")
        if lottie_airplane:
            st_lottie(lottie_airplane, height=120, key="airplane")

    with st.form("prediction_form"):
        st.subheader("Flight Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            airline = st.selectbox("Airline", df["airline"].unique())
            source_city = st.selectbox("Source City", df["source_city"].unique())
            departure_time = st.selectbox("Departure Time", df["departure_time"].unique())

        with col2:
            destination_city = st.selectbox("Destination City", df["destination_city"].unique())
            arrival_time = st.selectbox("Arrival Time", df["arrival_time"].unique())
            flight_class = st.selectbox("Class", df["class"].unique())

        with col3:
            stops = st.selectbox("Total Stops", df["stops"].unique())
            duration = st.number_input(
                "Flight Duration (Hours)",
                min_value=0.5,
                max_value=50.0,
                value=2.5,
                step=0.5,
            )
            days_left = st.number_input("Days Left Before Flight", min_value=1, max_value=100, value=10)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="🚀 Predict Ticket Price")

    if submit_button:
        with st.spinner("Analyzing data and generating prediction..."):
            try:
                model, encoder, scaler = load_models()

                # Format Inputs
                input_data = pd.DataFrame(
                    {
                        "airline": [airline],
                        "source_city": [source_city],
                        "departure_time": [departure_time],
                        "stops": [stops],
                        "arrival_time": [arrival_time],
                        "destination_city": [destination_city],
                        "class": [flight_class],
                        "duration": [duration],
                        "days_left": [days_left],
                    }
                )

                # Apply transformations (Log and Standard Scaler)
                input_data["duration"] = np.log1p(input_data["duration"])
                input_data["days_left"] = scaler.transform(input_data[["days_left"]])

                # Ordinal Encoding for Ordinal Features
                input_data["stops"] = input_data["stops"].map({"zero": 0, "one": 1, "two_or_more": 2})
                input_data["class"] = input_data["class"].map({"Economy": 0, "Business": 1})

                # Encode Categoricals
                object_cols = input_data.select_dtypes("object").columns
                encoded_cats = encoder.transform(input_data[object_cols])
                feature_names = encoder.get_feature_names_out(object_cols)

                input_data[feature_names] = encoded_cats
                input_data.drop(object_cols, axis=1, inplace=True)

                # Predict
                log_prediction = model.predict(input_data)
                final_price = np.expm1(log_prediction)[0]

                # Gamified Success
                st.balloons()
                st.toast("Prediction generated successfully!", icon="✅")

                # Professional Output Metric
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h2 style='color: gray; font-size: 20px;'>Estimated Ticket Price</h2>
                    <h1 style='color: #004aad; font-size: 45px;'>₹ {final_price:,.2f}</h1>
                    <p>Based on {airline} flights from {source_city} to {destination_city} ({flight_class} Class)</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Error making prediction. Please ensure models are in the 'models' folder. Details: {e}")

# Page 2: Dara Dashboard
elif page == "📊 Data Dashboard":
    st.title("📊 Interactive Data Dashboard")
    st.markdown("Explore the underlying data distributions and pricing trends.")

    # Using Tabs for a cleaner UI
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "💰 Price Analysis", "🗺️ Flight Routes"])

    with tab1:
        st.subheader("Airline Market Share & Hierarchy")
        # Professional Interactive Sunburst Chart
        fig_sunburst = px.sunburst(
            df,
            path=["class", "airline", "stops"],
            title="Flight Hierarchy (Class ➔ Airline ➔ Stops)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_sunburst.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig_sunburst, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Airlines Frequency**")
            fig, ax = plt.subplots(figsize=(8, 4))
            df["airline"].value_counts().sort_values().plot.barh(color="#004aad", ax=ax)
            ax.set_xlabel("Number of Flights")
            st.pyplot(fig)

        with col2:
            st.write("**Flights by Departure Time**")
            fig2 = px.histogram(
                df,
                x="departure_time",
                color="class",
                barmode="group",
                category_orders={
                    "departure_time": [
                        "Early_Morning",
                        "Morning",
                        "Afternoon",
                        "Evening",
                        "Night",
                        "Late_Night",
                    ]
                },
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Price Distributions & Outliers")
        st.markdown("Interactive boxplot to visualize the spread and outliers in ticket pricing.")

        # Interactive Boxplot inspired by your Data Processing.py
        fig_box = px.box(
            df,
            x="airline",
            y="price",
            color="class",
            title="Ticket Price Spread by Airline",
            labels={"price": "Price (INR)"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Price Timeline Trend")
        # Aggregating data to make the line chart readable
        trend_df = df.groupby(["days_left", "class"])["price"].mean().reset_index()
        fig_trend = px.line(
            trend_df,
            x="days_left",
            y="price",
            color="class",
            markers=True,
            title="Average Price vs. Days Left Before Departure",
            labels={"days_left": "Days Until Flight"},
        )
        fig_trend.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.subheader("City Connectivity")
        route_df = (
            df.groupby(["source_city", "destination_city"])
            .size()
            .reset_index(name="flight_count")
        )

        # Interactive Heatmap
        heatmap_data = route_df.pivot(index="source_city", columns="destination_city", values="flight_count").fillna(0)
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Flight Volume Between Cities",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# Page 3: About the developer
elif page == "👨‍💻 About Developer":
    st.title("👨‍💻 About the Developer")

    col1, col2 = st.columns([1, 2])
    with col1:
        # A nice developer animation
        lottie_dev = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_w51pcehl.json")
        if lottie_dev:
            st_lottie(lottie_dev, height=250)

    with col2:
        st.markdown(
            """
        ### **Abdallah Ahmad**
        **AI & ML Engineer**
        
        Second-year student at the **Faculty of Computer Science and Artificial Intelligence (FCAI), Cairo University**.
        
        #### **System Architecture**
        * **Data Engineering:** Missing value imputation, right-skew handling via logarithmic transformations (`np.log1p`), and feature scaling (`StandardScaler`).
        * **Feature Encoding:** One-Hot Encoding for nominal features (Airlines, Cities, Class).
        * **Machine Learning:** Evaluated Linear Regression, Random Forest, and XGBoost. Utilized `RandomizedSearchCV` for hyperparameter tuning.
        * **Deployment:** Built with Streamlit, Plotly for interactive dashboards, and cached resource loading for optimized web performance.
        """
        )