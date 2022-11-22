from flask import Flask,render_template,request
import os
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
from keras.models import load_model
import time
import gestures
from ibm_watson_machine_learning import APIClient 
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
import operator
import cv2 # opencv library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

bg = None

from tensorflow.keras.models import load_model#to load our trained model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder="templates", static_folder='static_files/') # initializing a flask app
# Loading the model
# model=load_model('new_train_model.h5')
# print("Loaded model from disk")
app_root = os.path.dirname(os.path.abspath(__file__))



wml_credentials = {
    "url" : "https://us-south.ml.cloud.ibm.com",
    "apikey" : "pRICfPi2WUbjuVJApM8QDMwsEDATYi3NKb_6pouqq4ed"
}
client = APIClient(wml_credentials)

def guid_from_space_name(client,space_name):
    space=client.spaces.get_details()
    return(next(item for item in space["resources"] if item["entity"]["name"]==space_name)["metadata"]["id"])

def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def _load_weights():
    try:
        # space_uid = guid_from_space_name(client, "gestureidentification")
        # print("Space UID=" + space_uid)
        # client.set.default_space(space_uid)
        # client.repository.download("ea40629b-9947-41c0-b0dc-123c51a6132f", "gesture-model.tgz")
        # from keras.models import load_model
        model = load_model("new_train_model_cloud.h5")
        return model
    except Exception as e:
        return None


def getPredictedClass(model):

    image = cv2.imread('static_files/Images/Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 120))

    gray_image = gray_image.reshape(1, 100, 120, 1)

    prediction = model.predict_on_batch(gray_image)

    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        return "Blank"
    elif predicted_class == 1:
        return "Ok"
    elif predicted_class == 2:
        return "Thumbs Up"
    elif predicted_class == 3:
        return "Thumbs Down"
    elif predicted_class == 4:
        return "Fist"
    elif predicted_class == 5:
        return "Five"   

@app.route('/', methods = ["GET","POST"])# route to display the home page
def hello():
    return render_template("intro.html")

@app.route('/about', methods = ["GET","POST"])# route to display the home page
def intro():
    return render_template("about.html")

@app.route('/predict', methods = ["GET","POST"])# route to display the home page
def predict():
    # OPTION 1 is get file input from user and manipulate
    if request.method == "POST":
        print("poost")
        target = os.path.join(app_root, 'static_files/Images/')
        f = request.files['filea']
        file_name = f.filename or ''
        destination = '/'.join([target, file_name])
        f.save(destination)
        print(destination)
    #     print(gest_pred)

    # OPTION 2 IS GET VIDEO INPUT and do operation
    # initialize accumulated weight
        accumWeight = 0.5

        # get the reference to the webcam
        camera = cv2.VideoCapture(0)
        # print(camera)

        fps = int(camera.get(cv2.CAP_PROP_FPS))
        # region of interest (ROI) coordinates
        top, right, bottom, left = 10, 350, 225, 590
        # initialize num of frames
        num_frames = 0
        # calibration indicator
        calibrated = False
        model = _load_weights()
        k = 0
        # keep looping, until interrupted
        while (True):
            # get the current frame
            (grabbed, frame) = camera.read()

            # resize the frame
            frame = imutils.resize(frame, width=700)
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated
            if num_frames < 30:
                run_avg(gray, accumWeight)
                if num_frames == 1:
                    print("[STATUS] please wait! calibrating...")
                elif num_frames == 29:
                    print("[STATUS] calibration successfull...")
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                    # count the number of fingers
                    # fingers = count(thresholded, segmented)
                    if k % (fps / 6) == 0:
                        cv2.imwrite('static_files/Images/Temp.png', thresholded)
                        predictedClass = getPredictedClass(model)
                        cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # show the thresholded image
                    cv2.imshow("Thesholded", thresholded)
            k = k + 1
            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break

        # free up memory
        camera.release()
        cv2.destroyAllWindows()
        print(predictedClass)
        print(destination)
        image1=cv2.imread(destination)
        if predictedClass == "Blank":       
            resized = cv2.resize(image1, (200, 200))
            cv2.imshow("Fixed Resizing", resized)
            print("fixed")
            cv2.imwrite('static_files/Images/Image_Operated.png', resized)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("1"):
                cv2.destroyWindow("Fixed Resizing")
        elif predictedClass== "Ok":   
            resized = cv2.rectangle(image1, (480, 170), (650, 420), (0, 0, 255), 2)
            cv2.imshow("Rectangle", image1)
            print("fixed")
            cv2.imwrite('static_files/Images/Image_Operated.png', resized)
            cv2.waitKey(0)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("0"):
                cv2.destroyWindow("Rectangle")         
        elif predictedClass=='Thumbs Up':
            print("Thumbs up")
            (h, w, d) = image1.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -45, 1.0)
            rotated = cv2.warpAffine(image1, M, (w, h))
            cv2.imshow("OpenCV Rotation", rotated)
            print("fixed")
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("2"):
                cv2.destroyWindow("OpenCV Rotation")
        elif predictedClass=='Thumbs Down':
            print("TD")
            blurred = cv2.GaussianBlur(image1, (21, 21), 0)
            cv2.imshow("Blurred", blurred)
            print("fixed")
            cv2.imwrite('static_files/Images/Image_Operated.png', blurred)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("3"):
                cv2.destroyWindow("Blurred")
        elif predictedClass=='Fist':
            print("Fist")
            resized = cv2.resize(image1, (400, 400))
            cv2.imshow("Fixed Resizing", resized)
            print("fixed")
            cv2.imwrite('static_files/Images/Image_Operated.png', resized)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("4"):
                cv2.destroyWindow("Fixed Resizing")
        elif predictedClass=='Five':
            print("5")
            gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            cv2.imshow("OpenCV Gray Scale", gray)
            print("fixed")
            cv2.imwrite('static_files/Images/Image_Operated.png', gray)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("5"):
                cv2.destroyWindow("OpenCV Gray Scale")

    return render_template('index.html')
   
if __name__ == "__main__":
   # running the app
    app.run(debug=True)