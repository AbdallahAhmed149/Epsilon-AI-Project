# ✈️ Kemet Flight Predictor (Epsilon-AI-Project)

**Kemet Flight Predictor** is a machine learning-powered web application designed to forecast airline ticket prices. By analyzing market trends, flight routes, and temporal data, the application provides accurate price estimates alongside an interactive data visualization dashboard to help users explore airline market dynamics.

-----

## ✨ Key Features

  * **🔮 Intelligent Price Predictor**: Enter your flight details (airline, source/destination, departure/arrival times, stops, class, and days left) to get an instant, highly accurate ticket price prediction powered by an optimized XGBoost model.
  * **📊 Interactive Data Dashboard**:
      * **📈 Market Overview**: Visualize airline market share and flight frequencies using Plotly sunburst charts and histograms.
      * **💰 Price Analysis**: Explore ticket price spread across airlines and track average price trends as the departure date approaches.
      * **🗺️ Flight Routes**: View flight volume between different cities through an interactive heatmap.
  * **⚡ Optimized Performance**: Features cached resource loading (`@st.cache_resource` and `@st.cache_data`) for lightning-fast model inference and seamless page navigation.
  * **🎨 Gamified UI**: Clean, modern interface with interactive Lottie animations and professional metric cards.

-----

## 🛠️ Tech Stack

  * **Frontend & UI**: [Streamlit](https://streamlit.io/), Streamlit-Lottie
  * **Data Manipulation**: Pandas, NumPy
  * **Data Visualization**: Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
  * **Machine Learning**: Scikit-Learn, XGBoost
  * **Model Serialization**: Joblib

-----

## 🧠 System Architecture & ML Pipeline

The prediction engine was built with a robust data engineering and machine learning pipeline:

1.  **Data Preprocessing**:
      * **Missing Values & Skewness**: Addressed right-skewed numerical features (like flight duration) using logarithmic transformations (`np.log1p`).
      * **Feature Scaling**: Applied `StandardScaler` to normalize continuous variables like the number of days left before the flight.
2.  **Feature Encoding**:
      * **Ordinal Encoding**: Mapped inherent hierarchies in features like the number of stops and flight class.
      * **One-Hot Encoding**: Transformed nominal categorical features (Airlines, Cities, Times) into machine-readable formats.
3.  **Model Selection & Tuning**:
      * Evaluated multiple algorithms including **Linear Regression**, **Random Forest**, and **XGBoost**.
      * Utilized `RandomizedSearchCV` for extensive hyperparameter tuning.
      * **Deployed Model**: The final XGBoost regressor was optimized with parameters such as `n_estimators=500`, `learning_rate=0.05`, and `max_depth=11`.

-----

## 🚀 Installation & Local Setup

Follow these steps to run the application on your local machine:

**1. Clone the repository**

```bash
git clone https://github.com/abdallahahmed149/epsilon-ai-project.git
cd epsilon-ai-project
```

**2. Install dependencies**
Make sure you have Python installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

*(Note: Ensure `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `plotly`, and `streamlit-lottie` are in your environment).*

**3. Ensure data and models are in place**
Verify that the following directories and files exist in your project folder:

  * `data/airlines_flights_data.csv`
  * `models/XGBoost.pkl`
  * `models/onehot_encoder.pkl`
  * `models/standard_scaler.pkl`

**4. Run the Streamlit app**

```bash
streamlit run app.py
```

-----

## 📁 Repository Structure

```text
📦 Epsilon-AI-Project
 ┣ 📂 data
 ┃ ┣ 📜 airlines_flights_data.csv    # Main dataset
 ┃ ┗ 📜 data_ml_model.csv            # Processed dataset
 ┣ 📂 imgs                           # Visualizations and EDA plots
 ┣ 📂 models
 ┃ ┣ 📜 XGBoost.pkl                  # Trained prediction model
 ┃ ┣ 📜 Random Forest.pkl            # Alternative trained model
 ┃ ┣ 📜 Linear Regression.pkl        # Baseline model
 ┃ ┣ 📜 onehot_encoder.pkl           # Fitted categorical encoder
 ┃ ┗ 📜 standard_scaler.pkl          # Fitted numerical scaler
 ┣ 📜 app.py                         # Main Streamlit application
 ┣ 📜 Data Processing.ipynb          # EDA and data cleaning notebook
 ┣ 📜 ML Model.ipynb                 # Model training and tuning notebook
 ┣ 📜 Data Processing.py             # Data processing scripts
 ┣ 📜 ML Model.py                    # Model training scripts
 ┣ 📜 README.md                      # Project documentation
 ┗ 📜 note.txt                       # Hyperparameter logs
```

-----

## 👨‍💻 About the Developer

**Abdallah Ahmad** *AI & ML Engineer*

Currently in the second year at the **Faculty of Computer Science and Artificial Intelligence (FCAI), Cairo University**, specializing in intelligent systems, machine learning architectures, and data engineering.

-----

*If you like this project, feel free to leave a ⭐ on the repository\!*
