import cv2
import time
import face_recognition
import pickle
import numpy as np
from pandas.conftest import datapath


def findEncodingsAndLocation(image):
    if image==None:
        return image
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faceLocation = face_recognition.face_locations(imageRGB, model="cnn")
    faceEncodings = face_recognition.face_encodings(imageRGB, faceLocation)

    return faceLocation, faceEncodings

def drawDetection(image, faceLocation):
    if image == None:
        return image
    top, right, bottom, left = faceLocation[0]
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

def findName (dataPath,faceEncodings,tolerance=0.4 ):
    with open(dataPath, 'rb') as file:
        employee_data = pickle.load(file)

    # Get the employee face encodings and IDs
    known_face_encodings = employee_data['encodings']
    known_face_ids = employee_data['names']

    for faceEncoding in faceEncodings:
        # Get if matches
        match = face_recognition.compare_faces(known_face_encodings, faceEncoding, tolerance)
        faceDistance = face_recognition.face_distance(known_face_encodings, faceEncoding)
        bestMatchIndex = np.argmin(faceDistance)
        name="unknown"
        if match[bestMatchIndex]:
            name = known_face_ids[bestMatchIndex]

        print(faceDistance, bestMatchIndex, match, name, match[bestMatchIndex])


cap = cv2.VideoCapture("Images/Benzima.jpg")
pTime = 0

while True:
    success, image = cap.read()

    if not success:
        break

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #enhance image
    #reducing noise by gauss blur
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.putText(image, f'FPS:{str(int(fps))}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)


    faceLocation, faceEncodings = findEncodingsAndLocation(image)
    # Draw rectangles around the detected faces
    drawDetection(image, faceLocation)
    findName ('employee.p',faceEncodings, tolerance=0.6)


    cv2.imshow("Image", image)
    cv2.waitKey(1)


