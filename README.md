# Face_recognition
This is a Face Recognition project written in Python.

The purpose of this program is to use create a model that is able to recognize the name of a certain person, given some pictures of the person itself that are used to train the model.

First there is the MCNN() face detector, which can find the faces inside a picture, and returns the rectangle containing them.
Then it uses a pretrained model, called FaceNet, to create the embeddings for each detected face.

The FaceNet output, with the corresponding labels, will be used to train a SVC model with a linear kernel that can then be used to make predictions on other pictures of which the subjects are unknown to the program. This time there can be more than one person inside the picture.

Finally the photos are plotted with a red rectangle containing the face and the relative predicted label.

Extra: if the probability related to the prediction is lower than a certain threshold, the predicted label will be '????'.

# How to use

## Required libraries
In order to make the program work correctly, you will need the following libraries, all easily installable via pip:
* numpy
* cv2
* sklearn
* tensorflow
* matplotlib
* mtcnn
* pickle



## Directories
You will just need to download the folder, already structured in the way the program wants it, which you can see here:

* FaceRecognition_Project
  * code_stuff
    * imports --> contains the FaceNet model I used
    * predictive_models --> contains the models you will save when training your program, so that you can access them without the need          to train it every time you just want to make some predictions
   * people ---> contains some subdirectories, named after the person you want to train your model to recognize, and each of these must contain images of the related person (make sure there is only one face for each pictures, otherwise it might give an error...)
   * to_predict ---> contains the images of which you want to detect and recognize the identity
      

## Last steps
After having put some pictures, both in the 'people' and the 'to_predict' folder, just copy the path to the main directory (in this case: FaceRecognition_Project), you can run your code!
