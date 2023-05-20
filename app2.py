import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras import preprocessing
import time
import pandas as pd
fig = plt.figure()

st.title('Leukemia Classifier')

st.markdown("Welcome to this simple web application that classifies Leukemia types. The types of Leukemia are ALL, AML, CLL, CML. The model here used is Ensemble model which is a combination of VGG 16, Resnet 50 and Inception V3")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                # st.subheader(predictions)
                data = {'Probability': ['> 50%','<50%'], 'Analysis': ['Belongs to this class particularly', 'May or may not belongs to this class']}
                df = pd.DataFrame(data)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(fig)

                with col2:
                    st.subheader(predictions)
                    st.dataframe(df)

def predict(image):
    classifier_model = "vgg_res_inc.hdf5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    img = image.resize((224,224))
    img = preprocessing.image.img_to_array(img)
    img = img/255
    # plt.imshow(img)
    classes = ['ALL','AML','CLL','CML']
    proba = model.predict(img.reshape(1,224,224,3))
    top_3 = np.argsort(proba[0])[:-4:-1]

    if((proba[0][top_3[0]])>0.50):
        result = f"This image belongs to {classes[top_3[0]]} with { (100 * proba[0][top_3[0]]).round(2) } % probability." 
    else:
        result = f"This image do not belong to any class"

    return result

    

if __name__ == "__main__":
    main()