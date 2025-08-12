import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model and class labels
model = load_model('model.h5')
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Image preprocessing
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array, img

# Prediction
def predict_image_class(img_array):
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class, prediction[0], predicted_class_index

# Streamlit UI
st.title("Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI image to classify the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, img = preprocess_image(uploaded_file)
    predicted_class, probabilities, index = predict_image_class(img_array)

    st.image(img, caption="Uploaded MRI", use_column_width=True)
    st.markdown(f"### Prediction: **{predicted_class}** ({probabilities[index]*100:.2f}%)")

    st.markdown("### Prediction Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"{class_labels[i]}: {prob * 100:.2f}%")
