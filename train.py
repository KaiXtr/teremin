import cv2 as cv
import numpy as np
from scipy import *

def carregarImagens():

    imagePaths = ('./archive')
    
    X = [] # Image data
    y = [] # Labels# Loops through imagepaths to load images and labels into arrays
    for path in imagePaths:
        img = cv.imread(path) # Reads image and returns np.array
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
        img = cv.resize(img, (320, 120)) # Reduce image size so training can be faster
        X.append(img)

        # Processing label in image path
        category = path.split("/")[3]
        label = int(category.split("_")[0][1]) # We need to convert 10_down to 00_down, or else it crashes
        y.append(label)# Turn X and y into np.array to speed up train_test_split

    X = np.array(X, dtype="uint8")
    X = X.reshape(len(imagePaths), 120, 320, 1) # Needed to reshape so CNN knows it's different images
    y = np.array(y)
    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    # Percentage of images that we want to use for testing. 
    # The rest is used for training.
    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

def construirModelo():
    # Construction of model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1))) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu')) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))# Configures the model for training
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])# Trains the model for a given number of epochs (iterations on a dataset) and validates it.
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))