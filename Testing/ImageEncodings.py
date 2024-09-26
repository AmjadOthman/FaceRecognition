import cv2
import face_recognition
import pickle
import os
import logging
import dlib

#set up logging
logging.basicConfig(level=logging.INFO)

#importing images
folderPath = "Images"
imagesPathList = os.listdir(folderPath)
imageList = []

#initialize known encodings and names
knownEncodings = []
knownNames = []

#split known images path to the id or name
#for i, imagePath in enumerate(imagesPathList):
#    #get the name
#    logging.info(f'Image {i} Processing: {imagePath}')
#    name = os.path.splitext(imagePath)[0]
#    #print(name)
#
#    #read the image and convert to RGB
#    image = cv2.imread(os.path.join(folderPath, imagePath))
#    image = cv2.GaussianBlur(image, (5, 5), 0)
#    if image is None:
#        logging.error(f"Error: Unable to read image file {imagePath}")
#        continue
#    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#    try:
#        #get image's encoding
#        #get face coordinate
#        logging.info(f'Image {i} encoding started...')
#        faceLocation = face_recognition.face_locations(imgRGB, model="cnn")
#        #get the encodings according to the face's location
#        imageEncoding = face_recognition.face_encodings(imgRGB, faceLocation)[0]
#        #print(f'name:{name}\nFace Location: {faceLocation}\n Encodings: {imageEncoding}')
#        logging.info(f'Image {i} encoding completed\n\n')
#    except Exception as e:
#        logging.error(f"Error: Unable to encode image {imagePath}: {e}")
#        continue
#
#    knownEncodings.append(imageEncoding)
#    knownNames.append(name)
#
#logging.info("serializing encodings...")
#data = {"encodings": knownEncodings, "names": knownNames}
#with open("employee.p", "wb") as f:
#    pickle.dump(data, f)
#logging.info("serializing encodings completed")
##print(knownNames, knownEncodings)
img = cv2.imread('Images/Ronaldo.jpg')

if img is None:
    print("Error: Unable to read image file")
else:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    dets = detector(imgRGB, 1)

    for k, d in enumerate(dets):
        x, y, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        face_aligned = imgRGB[y:y2, x:x2]

    cv2.imshow("Image", face_aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


