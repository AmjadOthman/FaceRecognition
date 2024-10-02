import cv2
import os
import pickle
import face_recognition
import logging
import dlib
import numpy as np

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
    if len(dets) > 0:
        return dets[0], imgGRAY
    else:
        return None, imgGRAY

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

def saveDataSet(knownEncodings, knownNames, file_path = "employee.p" ):
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

def loadData(file_path = "employee.p" ):
    """
    Desc:
        load the encodings with names in a pickle file.

    Args:
        file_path (str): Path to the pickle file (default: "employee.p")
    Return:
        the serialized encodings and names
    """
    with open(file_path, "rb") as f:
        data =  pickle.load(f)
        return data['encodings'], data['names']

def recognitionByImages():
    #set up logging
    logging.basicConfig(level=logging.INFO)

    #importing images
    folderPath = "Images"
    employees = os.listdir(folderPath)
    #print(imagesPathList)

    #initialize known encodings and names
    knownEncodings = []
    knownNames = []

    #split known images path to the id or name
    for i, employee in enumerate(employees):
        knownNames.append(employee)

        #get the directory list images
        employeeFolderPath = os.path.join(folderPath, employee)
        imagePathList = os.listdir(employeeFolderPath)

        imageEncodingList = []
        logging.info(f'{i+1}:{employee}: Images Processing and Encodings Started...')

        for i, imagePath in enumerate(imagePathList):
            imagePath = os.path.join(employeeFolderPath, imagePath)

            #read images
            image = cv2.imread(imagePath)

            if image is None:
                logging.error(f"Error: Unable to read image file {imagePath}")
                continue

            #get the detections of the face
            det, imgGRAY = getDets(image)
            x, y, x2, y2 = det.left(), det.top(), det.right(), det.bottom()

            #crop the face in the image
            face_aligned = imgGRAY[y:y2, x:x2]
            face_normalized = processFace(face_aligned)

            try:
                #get image's encoding, get face coordinate
                faceLocation = face_recognition.face_locations(face_normalized, model="cnn", number_of_times_to_upsample=1)
                imageEncoding = face_recognition.face_encodings(face_normalized, faceLocation, model="large", num_jitters=1)
                for encoding in imageEncoding:
                    imageEncodingList.append(encoding)

            except Exception as e:
                logging.error(f"Error: Unable to encode image {imagePath}: {e}")
                continue
        averageEncoding = np.average(imageEncodingList, axis=0)
        knownEncodings.append(averageEncoding)
        logging.info(f'{i}:{employee}: Images Processing and Encodings End.')

    #save data set in the file
    saveDataSet(knownEncodings, knownNames)

def recognnitionByVideo():
    cap = cv2.VideoCapture(0)
    names, encodings = loadData()
    encodingsList = []



    employeeName = input("Enter the name of Employer:\n")
    names.append(employeeName)


    mainFolderPath = "Images"
    employeeFolder = os.path.join(mainFolderPath, employeeName)

    if not os.path.exists(mainFolderPath):
        os.mkdir(employeeFolder)

    if not os.path.exists(employeeFolder):
        os.mkdir(employeeFolder)

    # Get the number of images in the employee's folder
    imagePathList = os.listdir(employeeFolder)
    imagePathList = [imagePath for imagePath in imagePathList if imagePath.endswith('.jpg') or imagePath.endswith('.png')]
    imageCounter = len(imagePathList)

    while True:
        success, frame = cap.read()

        if not success:
            logging.error("Unable to open camera")
            break

        print(f"Position the employee in direction {imageCounter} (e.g. front, left, right, etc.)")

        key = cv2.waitKey(0)

        if key == ord('c'):
            cv2.imwrite(f'{os.path.join(employeeFolder, str(imageCounter+1))}.jpg', frame)
            imageCounter+=1

            det, imgGRAY = getDets(frame)
            if det is not None:
                #crop the face
                x, y, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                #crop the face in the image
                face_aligned = imgGRAY[y:y2, x:x2]
                face_normalized = processFace(face_aligned)
                try:
                    #get image's encoding, get face coordinate
                    faceLocation = face_recognition.face_locations(face_normalized, model="cnn", number_of_times_to_upsample=1)
                    imageEncoding = face_recognition.face_encodings(face_normalized, faceLocation, model="large", num_jitters=1)[0]
                    encodingsList.append(imageEncoding)
                except Exception as e:
                    logging.error(f"Error: Unable to encode image {imageCounter+1}.jpg: {e}")
                    continue
            else:
                logging.error("Thee face is not detected!")
        elif key == ord('q'):
            break


        cv2.imshow("Video", frame)

    # Calculate the average encoding
    if len(encodingsList) > 0:
        averageEncoding = np.mean(encodingsList, axis=0)
        #encodings.append(averageEncoding)
        print("encodings",encodingsList)
        print(averageEncoding)
    else:
        logging.error("No encodings found")

    saveDataSet(encodings, names)



if __name__=="__main__":
    recognnitionByVideo()


