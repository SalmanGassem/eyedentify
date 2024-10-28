import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import time
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    page_title="Eyedentify", 
    page_icon=":eye:",
    layout="wide"
)

# Hide Streamlit default footer and main menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS to add margin above the navigation buttons in the sidebar
sidebar_navigation_margin = """
    <style>
    /* Target the navigation buttons in the sidebar */
    section[data-testid="stSidebar"] .css-1d391kg {
        margin-top: 40px;  /* Add margin to create space between the logo and the page buttons */
    }
    </style>
"""
st.markdown(sidebar_navigation_margin, unsafe_allow_html=True)

# Sidebar section
with st.sidebar:
    # Add logo at the top
    add_logo("app/logo4.png")
    st.caption("By Salman Gassem © 2024")

# Main page content (no changes here)
st.title(":eye: Eyedentify")
st.markdown("### Is your tyre ok?\n #### Find out by uploading an image of your tyre! :point_down:")
st.divider()

# File uploader section
supported_types = ["jpg", "jpeg", "png"]
uploaded_file = st.file_uploader("Upload an image:")

if uploaded_file is not None:
    file_type = uploaded_file.type.lower().split("/")[-1]
    if file_type not in supported_types:
        with st.status("Receiving image...", expanded=True) as status:
            st.write("Checking image type...")
            time.sleep(0.5)
        st.error(f"Unsupported file type '{file_type}'. Please upload JPG, JPEG, or PNG.")
    else:
        if uploaded_file:
            with st.status("Receiving image...", expanded=True) as status:
                st.write("Checking image type...")
                time.sleep(1)
                st.write("Confirmed!")
                time.sleep(1)
                st.write("Uploading...")
                time.sleep(1)
                status.update(label="Image received!", state="complete", expanded=False)

            time.sleep(1)

            # Model prediction logic
            if 'loaded_model' not in st.session_state:  # Check if model is already loaded
                loaded_model = tf.keras.models.load_model("app/savedmodel")
                st.session_state['loaded_model'] = loaded_model
            else:
                loaded_model = st.session_state['loaded_model']
                
            image = Image.open(uploaded_file)

            if image is not None:
                image = image.convert("RGB")
                resize = tf.image.resize(image, (256, 256))

            yhat = loaded_model.predict(np.expand_dims(resize/255, 0))

            st.divider()

            with st.spinner('Printing result...'):
                time.sleep(2)

            if yhat < 0.5:
                st.error('Tyre condition: Defective')
            else:
                st.success('Tyre condition: Good')

            st.image(image, caption="Uploaded Image", use_column_width=True)
