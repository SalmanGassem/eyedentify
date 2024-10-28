import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    page_title="Eyedentify", 
    page_icon=":eye:"
    )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

add_logo("app\logo4.png")

with st.sidebar:
      st.caption("By Salman Gassem © 2024")

with st.container():
    
    
    st.image("app\log_mid_border.png", use_column_width=True)

    st.header("About Eyedentify")
    st.write("Eyedentify is a project aimed at utilizing image classification techniques to address practical and meaningful challenges in various domains starting with industrial components quality control.")
    
    st.subheader("Why Eyedentify is the future")
    st.write("- Potential to revolutionize the way we solve problems.\n- Recognize and categorize objects, patterns, or features in images.\n- Automated disease diagnosis, object recognition in autonomous vehicles, and even facial recognition for security systems.\n- Machine learning algorithms and approaches that demonstrate accuracy and efficiency.")

    st.divider()

    st.subheader("Objective")
    st.write("Develop an Image Classification system capable of accurately identifying and categorizing manufacturing defects in factory-produced items.")

    st.subheader("To achieve")
    st.write("- Increased manufacturing profits and production quality.\n- Improve efficiency and reduce costs.\n- Enhance the manufacturer’s competitive advantage in the market.\n- Eliminate Human Error.\n- Reduce risk.")

