import tensorflow
from numpy.linalg import norm
# from tensorflow.keras.preprocessing import image
import os
import pickle
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D
from keras.applications import resnet
model= resnet.ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable= False
import numpy as np
model=  tensorflow.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

print(model.summary())


def extract_features(img_path,model):
    img= image.load_img(img_path,target_size=(224,224))
    img_array= image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=resnet.preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result= result/norm(result)
    return normalized_result
from tqdm import tqdm
filenames=[]
import os
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
print(len(filenames))
# print(os.listdir('images'))

feature_list= []
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))