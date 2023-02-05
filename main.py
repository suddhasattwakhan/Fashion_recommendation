import streamlit as st
import os
from PIL import Image
from numpy.linalg import norm
from keras.applications import resnet
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import tensorflow
feature_list= np.array(pickle.load(open('embeddings.pkl','rb')))

filenames=pickle.load(open('filenames.pkl','rb'))

model= resnet.ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable= False
import numpy as np
model=  tensorflow.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

st.title('Fashion recommender system')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
def feature_extraction(img_path,model):
    img= image.load_img(img_path,target_size=(224,224))
    img_array= image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=resnet.preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result= result/norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors= NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    distances,indices= neighbors.kneighbors([features])
    return indices


uploaded_file= st.file_uploader('choose an image')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image=Image.open(uploaded_file)
        st.image(display_image)
        features=feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        # st.text(features)
        indices= recommend(features,feature_list)
        col1,col2,col3,col4,col5= st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")


