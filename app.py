import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import base64

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Food Delivery Time Prediction", page_icon="🍔", layout="centered")


# 🎨 Background Styling
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    bg_image = f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Call the function
set_background("p8.jpg")




# ---------------- Top Card Section ----------------
st.markdown(
    """
    <div style="
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    ">
        <h1 style="color: green; font-size: 2em;">
            🥗 Fresh, Fast & On-Time
        </h1>
        <p style="font-size: 1.2em; color: #333; margin-bottom: 30px;">
            "Predicting food delivery times with intelligence — because freshness can’t wait!"
        </p>
        <a href="#predict-section">
            <button style="
                background-color: #d94600;
                color: white;
                font-size: 1.1em;
                font-weight: bold;
                padding: 12px 30px;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            ">
                🚀 Predict Now
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Full-Page Background ----------------
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
        padding: 0;
        background: url('p4.png') no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Landscape Image ----------------
st.image("p4.png", use_container_width=True)  # automatically fits width of page
st.markdown("<p style='text-align: center; color: gray;'>Delicious meals delivered fresh 🍴</p>", unsafe_allow_html=True)

# ---------------- Prediction Section ----------------
st.markdown('<div id="predict-section"></div>', unsafe_allow_html=True)
st.markdown("## Enter details to predict delivery time ⏳")

# ---------------- Load Model ----------------
model = load_model("food_delivery_model.keras")


import streamlit as st

# Sidebar content
with st.sidebar:
    # Navigation header
    st.markdown("### 🔍 **Navigation**")

    # Home button
    if st.button("🏠 Home"):
        st.rerun()

    # Link button
    st.link_button("🌍 Haversine formula", "https://en.wikipedia.org/wiki/Haversine_formula")

    # Divider
    st.markdown("---")

    # How it works section
    st.markdown("""
    ### 🔍 How it works:
    
    **Data Preparation & Cleaning**  
    - Reads delivery data  
    - Cleans missing values and extracts the target column **`Time_taken(min)`**  
    - Calculates **distance between restaurant and customer** using the Haversine formula.  
    
    **Model Training**  
    - Uses selected features:  
      - 🚴 Delivery partner's **age**  
      - ⭐ Delivery partner's **ratings**  
      - 📍 Calculated **distance (km)**  
    - Trains a **LSTM neural network** to learn delivery patterns
    """)

    # Divider
    st.markdown("---")

    # Navigation buttons
    if st.button("📈 Exploratory Data Analysis"):
        st.switch_page("pages/eda.py")

   




# ---------------- Input Fields ----------------
input_age = st.number_input("Age of Delivery Partner", min_value=18, max_value=65, step=1, value=25)
input_rating = st.number_input("Ratings of Previous Deliveries", min_value=1.0, max_value=5.0, step=0.1, value=4.5)
input_distance = st.number_input("Total Distance (km)", min_value=0.1, step=0.1, value=5.0)

# Predict button
if st.button("Predict Now", key="main_predict"):
    try:
        features = np.array([[input_age, input_rating, input_distance]])
        features = features.reshape((features.shape[0], features.shape[1], 1))
        prediction = model.predict(features, verbose=0)
        result = round(float(prediction[0][0]), 2)
        st.success(f"⏱️ Predicted Delivery Time: {result} minutes")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------------- Demo Examples ----------------
st.subheader("📌 Quick Demo Examples")
demo_examples = [
    {"title": "Example 1: Young, high-rated, short distance", "Age": 25, "Rating": 4.5, "Distance": 3.2},
    {"title": "Example 2: Experienced, medium distance", "Age": 35, "Rating": 3.8, "Distance": 8.5},
    {"title": "Example 3: Experienced, long distance", "Age": 45, "Rating": 4.2, "Distance": 15.0},
]

for idx, demo in enumerate(demo_examples, start=1):
    st.markdown(f"### {demo['title']}")
    st.write(f"- 👤 Age: {demo['Age']}\n- ⭐ Rating: {demo['Rating']}\n- 📍 Distance: {demo['Distance']} km")
    if st.button(f"Predict #{idx}"):
        try:
            f = np.array([[demo["Age"], demo["Rating"], demo["Distance"]]])
            f = f.reshape((f.shape[0], f.shape[1], 1))
            pred = round(float(model.predict(f, verbose=0)[0][0]), 2)
            st.success(f"⏱️ Predicted Delivery Time: {pred} minutes")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Sidebar custom style (dark wood brown)
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color:#4B2E20; /* Dark brown background */
        }
        [data-testid="stSidebar"] * {
            color: white; /* Make all text white for contrast */
        }
    </style>
    """,
    unsafe_allow_html=True
)