import cv2
import os
import pickle
import face_recognition
import logging
import dlib

def getDets(image):
    """
    Desc:
        Convert image to grayscale and apply face alignment detector.

    Args:
        image (numpy array): Input image

    Returns:
        dets (list): Face detections
        imgGRAY (numpy array): Grayscale image
    """
    imgGRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    dets = detector(imgGRAY,1)
    return dets, imgGRAY

def processFace(face_aligned):
    """
    Desc:
        Apply histogram equalization, Gaussian filter, and normalization to the face image.

    Args:
        face_aligned (numpy array): Aligned face image

    Returns:
        face_normalized (numpy array): Normalized face image
    """
    face_eq = cv2.equalizeHist(face_aligned)
    face_filtered = cv2.GaussianBlur(face_eq, (5, 5), 0)
    face_normalized = cv2.resize(face_filtered, (112, 112))
    face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_GRAY2RGB)

    return face_normalized

def saveDataSet(knownEncodings, knownNames, file_path ="employee.p" ):
    """
    Desc:
        Serialize the encodings with names in a pickle file.

    Args:
        knownEncodings (list): List of face encodings
        knownNames (list): List of corresponding names
        file_path (str): Path to the pickle file (default: "employee.p")
    """
    logging.info("serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    logging.info("serializing encodings completed")

def main():
    #set up logging
    logging.basicConfig(level=logging.INFO)

    #importing images
    folderPath = "Images"
    imagesPathList = os.listdir(folderPath)

    #initialize known encodings and names
    knownEncodings = []
    knownNames = []

    #split known images path to the id or name
    for i, imagePath in enumerate(imagesPathList):

        #get the name
        logging.info(f'Image {i} Processing: {imagePath}')
        name = os.path.splitext(imagePath)[0]

        #read the image and convert to RGB
        image = cv2.imread(os.path.join(folderPath, imagePath))

        if image is None:
            logging.error(f"Error: Unable to read image file {imagePath}")
            continue

        #get face detections
        dets, imgGRAY = getDets(image)

        print("dets", dets)
        for k, d in enumerate(dets):
            x, y, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            face_aligned = imgGRAY[y:y2, x:x2]

            face_normalized = processFace(face_aligned)

        try:
            #get image's encoding, get face coordinate
            logging.info(f'Image {i} encoding started...')
            faceLocation = face_recognition.face_locations(face_normalized, model="cnn", number_of_times_to_upsample=1)
            imageEncoding = face_recognition.face_encodings(face_normalized, faceLocation, model="large", num_jitters=1)[0]
            knownEncodings.append(imageEncoding)
            knownNames.append(name)
            print(f'name:{name}\nFace Location: {faceLocation}\n Encodings: {imageEncoding}')
            #save encodings and names

            logging.info(f'Image {i} encoding completed\n\n')
        except Exception as e:
            logging.error(f"Error: Unable to encode image {imagePath}: {e}")
            continue


    #save data set in the file
    saveDataSet(knownEncodings, knownNames)

    #test the pickle file
    #f=open("employee.p", "rb")
    #data = pickle.load(f)
    #print(data)
    #f.close()




if __name__=="__main__":
    main()


