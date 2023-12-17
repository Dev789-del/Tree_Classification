import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import glob
from tkinter import *
from PIL import Image, ImageTk
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os
import shutil

class Tree_Data:  
    def load_image():
        #Function to load image
        data_image = []
        for filename in glob.glob('./images/*.jpg'):
            image = Image.open(filename)
            data_image.append(image)
        print("Image loaded successfully")
        
        for image in data_image:
            plt.imshow(image)
            plt.show()

    def train_image():
        #Show 3 random train images
        train_image = []
        for filename in glob.glob('./new_dataset/train/*/*.png'):
            image = Image.open(filename)
            train_image.append(image)
        for image in train_image[:3]:
            plt.imshow(image)
            plt.show()
        
    def test_image():
        #Show 3 random test images
        test_image = []
        for filename in glob.glob('./new_dataset/test/*.png'):
            image = Image.open(filename)
            test_image.append(image)
        for image in test_image[:3]:
            plt.imshow(image)
            plt.show()

    def make_train_csv():
        #Delete train csv file if it exists
        if os.path.exists('./model/train.csv'):
            os.remove('./model/train.csv')
        #Define array to store image data   
        image_names = []
        image_labels = []
        image_heights = []
        image_widths = []
        #Get image names and tree names, tree sizes in n x n format
        for filename in glob.glob('./new_dataset/train/*/*.png'):
            image_names.append(filename.split('\\')[-1])
            image_labels.append(filename.split('\\')[-2])
            for image in glob.glob(filename):
                image = cv2.imread(image)
                image_heights.append(image.shape[0])
                image_widths.append(image.shape[1])

        #Create dataframe
        df = pd.DataFrame()
        df['image_names'] = image_names
        df['image_labels'] = image_labels
        df['image_heights'] = image_heights
        df['image_widths'] = image_widths
        df.to_csv('./model/train.csv', index=False)
    def generate_tree_image():
        #Delete all folders and files in new_dataset folder
        folder = './new_dataset/train/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        #Function to generate 10 tree data images from images folder into train folder of new_dataset
        data_image = []
        for filename in glob.glob('./images/*.jpg'):
            image = Image.open(filename)
            data_image.append(image)
        #Get image names from example images in images folder
        image_names = []
        for filename in glob.glob('./images/*.jpg'):
            image_names.append(filename.split('\\')[-1].split('.')[0])
        #Sort image names from A to Z based on first letter
        image_names_sort = sorted(image_names)
        #Check if folder exists with name in image_names, if not, create folder
        for name in image_names:
            if not os.path.exists('./new_dataset/train/' + str(name)):
                os.makedirs('./new_dataset/train/' + str(name))
        for i in range(50):
            for image, name in zip(data_image, image_names_sort):
                image = image.resize((299, 299))
                image = np.array(image)
                image = image.reshape((1, 299, 299, 3))
                datagen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
            #Save the generated images into tree name folder
                for batch in datagen.flow(image, batch_size=1, save_to_dir='./new_dataset/train/' + str(name), save_prefix= name, save_format='png'):
                    break

    def generate_test_image():
        #Delete all folders and files in new_dataset folder
        folder = './new_dataset/test/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        #Make test random images from images folder
        test_image = []
        for filename in glob.glob('./images/*.jpg'):
            image = Image.open(filename)
            test_image.append(image)
        #Generate 10 random images from images folder
        for i in range(10):
            for image in test_image:
                image = image.resize((299, 299))
                image = np.array(image)
                image = image.reshape((1, 299, 299, 3))
                datagen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
                for batch in datagen.flow(image, batch_size=1, save_to_dir='./new_dataset/test', save_prefix= 'test', save_format='png'):
                    break
        
    def make_test_csv():
        #Delete test csv file if it exists
        if os.path.exists('./model/test.csv'):
            os.remove('./model/test.csv')
        #Define array to store image data   
        image_names = []
        image_heights = []
        image_widths = []
        #Get image names and tree names, tree sizes in n x n format
        for filename in glob.glob('./new_dataset/test/*.png'):
            image_names.append(filename.split('\\')[-1])
            for image in glob.glob(filename):
                image = cv2.imread(image)
                image_heights.append(image.shape[0])
                image_widths.append(image.shape[1])

        #Create dataframe
        df = pd.DataFrame()
        df['image_names'] = image_names
        df['image_heights'] = image_heights
        df['image_widths'] = image_widths
        df.to_csv('./model/test.csv', index=False)
#Set parameters
batch_size = 50 # Number of images to be considered in each batch
epochs = 10 # Number of times the entire dataset is passed through the network
num_classes = 10

def load_data():
    #Split data into train and test
    train_data = pd.read_csv('./model/train.csv')
    test_data = pd.read_csv('./model/test.csv')

    #Get train data
    X_train = []
    y_train = []
    for filename, label in zip(train_data['image_names'], train_data['image_labels']):
        #Fix module cv2 has no attribute imread
        image = cv2.imread('./new_dataset/train/' + label + '/' + filename)
        image = cv2.resize(image, (299, 299))
        X_train.append(image)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #Get test data
    X_test = []
    for filename in test_data['image_names']:
        image = cv2.imread('./new_dataset/test/' + filename)
        image = cv2.resize(image, (299, 299))
        X_test.append(image)

    X_test = np.array(X_test)

    #Convert labels to one-hot vectors
    y_train = pd.get_dummies(y_train)
    y_train = np.array(y_train)

    #Normalize data
    X_train = X_train.astype('float32')
    X_train /= 255
    X_test = X_test.astype('float32')
    X_test /= 255

    #Define y_test
    y_test = pd.read_csv('./model/test.csv')
    y_test = np.array(y_test['image_names'])
    return X_train, y_train, X_test, y_test

def build_model():

    #Build model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(299, 299, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5, activation='softmax'))

    #Save model
    model.save('./model/project_model.h5')

#Evaluate model and fix ValueError: Shapes (50, 5) and (50, 1) are incompatible
def evaluate_model(model, X_train, y_train, X_test, y_test):
    #Compile model
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    #Fit model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    #Evaluate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

#Make prediction on 1 test image
def predict_image(model, X_test, y_test):
    #Get random image from test data
    random_image = np.random.randint(0, len(X_test))
    image = X_test[random_image]
    image = image.reshape((1, 299, 299, 3))
    #Predict image
    prediction = model.predict(image)
    #Get label of predicted image
    prediction = np.argmax(prediction)
    #Get label of actual image
    actual = y_test[random_image]
    #Get image name
    image_name = y_test[random_image]
    #Print actual and predicted label
    print('Actual label: ', actual)
    print('Predicted label: ', prediction)
    #Show image
    image = cv2.imread('./new_dataset/test/' + image_name)
    plt.imshow(image)
    plt.show()


#Describe the csv file
def describe_csv():
    #Load data from train.csv
    train_data = pd.read_csv('./model/train.csv')
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data.head()

    #Describe the csv file
    print(train_data.describe())
    print(train_data.info())

#Run describe_csv function
# describe_csv()

# Run test_image function
# Tree_Data.generate_test_image()

# Run make_test_csv function
# Tree_Data.make_test_csv()

# Describe the test csv file
# test_data = pd.read_csv('./model/test.csv')
# print(test_data.describe())
# #Run load_data function
X_train, y_train, X_test, y_test = load_data()
# print(X_train.shape)

# Delete existing model to refresh
# if os.path.exists('./model/project_model.h5'):
#     os.remove('./model/project_model.h5')
# Make model 
# build_model()

# Run evaluate_model function
# model = tf.keras.models.load_model('./model/project_model.h5')
# evaluate_model(model, X_train, y_train, X_test, y_test)

# Show train image and test image
# Tree_Data.train_image()
# Tree_Data.test_image()