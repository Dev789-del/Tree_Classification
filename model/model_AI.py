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
from tkinter import filedialog
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

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

    def generate_train_image():
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
        #Define the image data generator
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=True,
            horizontal_flip=True,
            fill_mode='nearest')
        #Get the list of only tree folders names in images folder(other file not folder in images folder will be ignored)
        tree_names = [tree_name for tree_name in os.listdir('./images') if os.path.isdir(os.path.join('./images', tree_name))]
        tree_names = sorted(tree_names)
        #Create folder with name as tree type in train folder of new_dataset
        for tree_name in tree_names:
            if not os.path.exists('./new_dataset/train/' + tree_name):
                os.makedirs('./new_dataset/train/' + tree_name)
        #In each tree folder, read all images and generate 10 images per image in it and save to the same tree type folder in train folder of new_dataset
        for tree_name in tree_names:
            for filename in glob.glob('./images/' + tree_name + '/*.jpg'):
                image = Image.open(filename)
                image = image.resize((299, 299))
                image = np.array(image)
                image = image.reshape((1, 299, 299, 3))
                # Print the filename and the shape of the image
                print(f"Processing {filename} with shape {image.shape}")
                try:
                    image = image.reshape((1, 299, 299, 3))
                except ValueError as e:
                    print(f"Cannot reshape image {filename} with shape {image.shape}")
                    continue
                i = 0
                for batch in datagen.flow(image, batch_size=1, save_to_dir='./new_dataset/train/' + tree_name, save_prefix=tree_name, save_format='png'):
                    i += 1
                    if i > 10:
                        break
    def make_train_csv():
        #Delete train csv file if it exists
        if os.path.exists('./model/train.csv'):
            os.remove('./model/train.csv')
        #Define array to store image data   
        image_names = []
        image_labels = []
        image_heights = []
        image_widths = []
        #Get tree names and set image label
        for filename in glob.glob('./new_dataset/train/*/*.png'):
            image_names.append(filename.split('\\')[-1])
            image_labels.append(filename.split('\\')[-1].split('.')[0])
            image = Image.open(filename)
            image_heights.append(image.height)
            image_widths.append(image.width)
        #Create dataframe
        df = pd.DataFrame()
        df['image_names'] = image_names
        df['image_labels'] = image_labels
        df['image_heights'] = image_heights
        df['image_widths'] = image_widths
        df.to_csv('./model/train.csv', index = False)

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
        #Define the image data generator
        datagen = ImageDataGenerator(
            rotation_range=10,
            vertical_flip=True,
            horizontal_flip=True,
            fill_mode='nearest')
        #Get the list of only tree folders names in images folder(other file not folder in images folder will be ignored)
        tree_names = [tree_name for tree_name in os.listdir('./images') if os.path.isdir(os.path.join('./images', tree_name))]
        tree_names = sorted(tree_names)
        # Initialize the counter for the tree folders
        folder_counter = 0

        # For each tree folder in the images folder
        for tree_name in tree_names:
            # If the counter is 10, break the loop
            if folder_counter == 20:
                break
            # Create a new folder in the test folder for the generated images
            if not os.path.exists('./new_dataset/test/' + str(folder_counter + 1)):
                os.makedirs('./new_dataset/test/' + str(folder_counter + 1))
            # For each image in the tree folder
            for filename in glob.glob('./images/' + tree_name + '/*.jpg'):
                image = Image.open(filename)
                image = image.resize((299, 299))
                image = np.array(image)
                image = image.reshape((1, 299, 299, 3))

                # Generate 2 new images from the original image
                i = 0
                for batch in datagen.flow(image, batch_size=1, save_to_dir='./new_dataset/test/' + str(folder_counter + 1), save_prefix= folder_counter + 1, save_format='png'):
                    i += 1
                    if i > 2:
                        break

            # Increment the counter
            folder_counter += 1
    
    def make_test_csv():
        #Delete test csv file if it exists
        if os.path.exists('./model/test.csv'):
            os.remove('./model/test.csv')
        #Define array to store image data   
        image_names = []
        image_labels = []
        
        #Get image names and image labels based on new generated test images function
        for filename in glob.glob('./new_dataset/test/*/*.png'):
            image_names.append(filename.split('\\')[-1])
            image_labels.append(filename.split('\\')[-2])
        #Create dataframe
        df = pd.DataFrame()
        df['image_names'] = image_names
        df['image_labels'] = image_labels
        df.to_csv('./model/test.csv', index = False)
#Set parameters
batch_size = 54
epochs = 10

def load_data():
    #Split data into X_train, y_train, X_test, y_test
    train_data = pd.read_csv('./model/train.csv')
    test_data = pd.read_csv('./model/test.csv')
    #Define X_train, X_test with image from train data and test data
    X_train = []
    X_test = []
    for filename in glob.glob('./new_dataset/train/*/*.png'):
        image = Image.open(filename)
        X_train.append(image)
    for filename in glob.glob('./new_dataset/test/*/*.png'):
        image = Image.open(filename)
        X_test.append(image)
    #Define y_train, y_test with image labels from train data and test data
    y_train = train_data['image_labels']
    y_test = test_data['image_labels']
    #Convert X_train, X_test to numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    #Normalize X_train, X_test
    X_train = X_train / 255
    X_test = X_test / 255
    #Return X_train, y_train, X_test, y_test
    return X_train, y_train, X_test, y_test

def build_model():
    #Build model
    model = Sequential()

    #first layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(299, 299, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    #second layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #third layer
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    #fourth layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #fifth layer
    model.add(Flatten())

    #sixth layer
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))

    #Save model
    model.save('./model/project_model.h5')

#Evaluate model and fix ValueError: Shapes (None, 1) and (None, 10) are incompatible
def evaluate_model(model, X_train, y_train, X_test, y_test):
    #Encode y_train, y_test
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    #Convert y_train, y_test to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #Compile model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #Fit model
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
    #Evaluate model
    score = model.evaluate(X_test, y_test)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    #Save model
    model.save('./model/project_model.h5')
    
def make_validation_csv():
    #Delete test csv file if it exists
    if os.path.exists('./model/validation.csv'):
        os.remove('./model/validation.csv')
    #Define tree names and tree order based on A to Z in images folder
    tree_names = [tree_name for tree_name in os.listdir('./images') if os.path.isdir(os.path.join('./images', tree_name))]
    tree_names = sorted(tree_names) 
    tree_order = [i for i in range(1, len(tree_names) + 1)]
    #Create dataframe
    df = pd.DataFrame()
    df['image_label'] = tree_order
    df['tree_name'] = tree_names
    df.to_csv('./model/validation.csv', index = False)
 
#Predict a random image and show predicted label, actual label and image name 
def predict_image(model):
    # Open a file chooser dialog and allow the user to select an input in the test folder
    path = filedialog.askopenfilename(initialdir='./test', title='Select an image')
    # Get the name of the image
    image_name = os.path.basename(path)
        
    # Ensure a file path was selected
    if len(path) > 0:
        #Read image 
        image = Image.open(path)
        image = image.resize((299, 299))
        image = np.array(image)
        image = image.reshape((1, 299, 299, 3))
        #Predict image based on model 
        predicted_label = model.predict(image)
        #Get index of predicted label
        predicted_label = np.argmax(predicted_label)
        #Load data from validation.csv
        validation_data = pd.read_csv('./model/validation.csv')
        #Convert predicted label to tree name in validation.csv by using predicted label as index
        predicted_label = validation_data['image_label'][int(predicted_label)]
        #Get actual label from folder name of selected image
        actual_label = path.split('/')[-2]
        #Convert actual label to tree name in validation.csv by using actual label as index
        actual_label = validation_data['image_label'][int(actual_label)-1]
        #Show predicted label, actual label and image name
        print("Predicted label: ", predicted_label)
        print("Actual label: ", actual_label)
        print("Image name: ", image_name)
        # Show image
        plt.imshow(image.reshape(299, 299, 3))
        plt.show()
        
# Generate train image and test image
# Tree_Data.generate_train_image()
# Tree_Data.generate_test_image()
# Make train csv file and test csv file
Tree_Data.make_train_csv()
Tree_Data.make_test_csv()
# Make validation csv file
# make_validation_csv()
#Load data
# X_train, y_train, X_test, y_test = load_data()
# Delete existing model to refresh
# if os.path.exists('./model/project_model.h5'):
    # os.remove('./model/project_model.h5')
#  Make model 
# build_model()
# Run evaluate_model function
# model = tf.keras.models.load_model('./model/project_model.h5')
# evaluate_model(model, X_train, y_train, X_test, y_test)
# predict_image(model)
# Show train image and test image
# Tree_Data.train_image()
# Tree_Data.test_image()