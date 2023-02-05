import pickle
import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications import resnet
from numpy.linalg import norm
import tensorflow 
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list= np.array(pickle.load(open('embeddings.pkl','rb')))

filenames=pickle.load(open('filenames.pkl','rb'))

model= resnet.ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable= False
import numpy as np
model=  tensorflow.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])


img= image.load_img('sample/shirt.webp',target_size=(224,224))
img_array= image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
preprocessed_img=resnet.preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result= result/norm(result)

neighbors= NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances,indices= neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',temp_img)
    cv2.waitKey(0)
