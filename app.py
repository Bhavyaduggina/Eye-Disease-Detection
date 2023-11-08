import streamlit as st
from keras.models import load_model  
import os
from PIL import Image,ImageOps 
import numpy as np
from streamlit_lottie import st_lottie
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.set_page_config(page_title="Eye Disease Detector",page_icon="🔎👁️",layout="wide")
def lottie_file(filepath: str):
    with open(filepath,'r') as f:
        return json.load(f)
st.markdown("<h2 style='text-align: center; color: #2E8B57;'>Eye Disease Detection using Deep Learning 👁️🔍</h2>", unsafe_allow_html=True)
st.markdown("---")
st.subheader("Vision Insight: Automated Diagnosis of Diabetic Retinopathy, Cataract, Glaucoma, and Normal Vision using Advanced Deep Learning Technology")
lottie_pic=lottie_file("C:/Users/91741/Downloads/Eye Disease Prediction/Animation - 1699444623556.json")
st_lottie(
      lottie_pic,
      speed=1,
      reverse=False,
      loop=True,
      height=600,
      width=2000,
      key='Hi' )

model = load_model('model.h5')

st.text("Please provide an EYE Image for Analysis.")
uploaded_file = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=540)
    st.write("Classifying...")
    lottie_pic=lottie_file("C:/Users/91741/Downloads/Eye Disease Prediction/Animation - 1699444321307.json")
    st_lottie(
      lottie_pic,
      speed=1,
      reverse=False,
      loop=True,
      height=600,
      width=2000,
      key='Hello'
    )
    st.write("")
    img = image.convert('RGB')  # Convert PIL image to RGB format
    img = img.resize((224, 224))  # Adjust the dimensions as per your model's input requirements

    # Normalize the image data
    img = np.array(img)
    img = img / 255.0  # Assuming your model expects input in the range [0, 1]

    # Expand dimensions to create a batch of one
    img = np.expand_dims(img, axis=0)

    # Perform the prediction
    prediction = model.predict(img)
    yclass = np.argmax(prediction, axis=1)
    if yclass == 0:
        st.subheader("The patient has been diagnosed with Diabetic Retinopathy, a condition associated with diabetes that affects the blood vessels in the retina.")
    elif yclass == 1:
        st.subheader("The patient has been diagnosed with Cataract, a common eye condition that causes clouding of the eye's lens, leading to blurred vision.")
    elif yclass == 2:
        st.subheader("No significant eye disease has been detected.")
    elif yclass == 3:
        st.subheader("The patient has been diagnosed with Glaucoma, a group of eye conditions that can cause blindness by damaging the optic nerve.")
    else:
        st.subheader("Invalid classification.")



st.markdown("<p style='text-align: center; color: green; font-size: 25px; margin-top: 50px;'>DEVELOPED BY - BHAVYA,DUSHYANTH,DHARANI,HARSHITH</p>", unsafe_allow_html=True)