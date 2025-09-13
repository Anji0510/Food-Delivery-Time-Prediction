import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import base64

st.set_page_config(page_title="EDA - Food Delivery", page_icon="📊", layout="wide")

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
set_background("p7.jpg")

@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    data['Time_taken(min)'] = data['Time_taken(min)'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
    data['Time_taken(min)'] = pd.to_numeric(data['Time_taken(min)'], errors='coerce')
    numeric_columns = ['Delivery_person_Age', 'Delivery_person_Ratings',
                       'Restaurant_latitude', 'Restaurant_longitude',
                       'Delivery_location_latitude', 'Delivery_location_longitude']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=numeric_columns + ['Time_taken(min)'])

    # Distance calculation
    R = 6371
    def deg_to_rad(deg): return deg * (np.pi/180)
    def distcalc(lat1, lon1, lat2, lon2):
        dlat, dlon = deg_to_rad(lat2-lat1), deg_to_rad(lon2-lon1)
        a = np.sin(dlat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(dlon/2)**2
        return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
    data["distance"] = data.apply(lambda row: distcalc(row['Restaurant_latitude'], row['Restaurant_longitude'],
                                                       row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)
    return data

data = load_data()

st.title("📊 Exploratory Data Analysis")
st.write("This app predicts *Food Delivery Time* using ML (LSTM).")

st.subheader("📝Dataset Overview")

st.write(f"**Final dataset shape:** {data.shape}")
st.dataframe(data.head())

st.subheader("📈Plots")

# ================================
# Updated Plots with Light Theme
# ================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Distance vs Time", "Age vs Time", "Ratings vs Time", "Boxplot (Vehicle/Order)"
])

with tab1:
    fig1 = px.scatter(
        data,
        x="distance",
        y="Time_taken(min)",
        trendline="ols",
        title="Distance vs Time Taken"
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.scatter(
        data,
        x="Delivery_person_Age",
        y="Time_taken(min)",
        size="distance",
        color="distance",
        color_continuous_scale=["navy", "purple", "magenta", "orange", "yellow"],
        trendline="ols",
        title="Age vs Time Taken"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.scatter(
        data,
        x="Delivery_person_Ratings",
        y="Time_taken(min)",
        size="distance",
        color="distance",
        trendline="ols",
        title="Ratings vs Time Taken"
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    if 'Type_of_vehicle' in data.columns and 'Type_of_order' in data.columns:
        fig4 = px.box(
            data,
            x="Type_of_vehicle",
            y="Time_taken(min)",
            color="Type_of_order",
            title="Delivery Time by Vehicle & Order Type"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.write("🚘 Vehicle/Order type columns not found in dataset.")

#text


st.sidebar.markdown("---")
st.sidebar.markdown(""" 
### 🔍 How it works:
 **(EDA)**  
   - Explore trends like:
     **distance vs. time**, 
     **age vs. time**,      
     **ratings vs. time**,
     **vehicle/order type vs. time**.
            

 **Prediction**  
   - Input values (age, ratings, distance).  
   - Get an **instant prediction** of delivery time in **minutes**, with helpful 
     context (fast, standard, or delayed).  
""")


st.sidebar.markdown("---")


# Sidebar Navigation

if st.sidebar.button(" Back "):
    st.switch_page("app.py")  # Navigates back to main.py 

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

st.write("Thank you for visiting our App! 😊",layout ="centered")
