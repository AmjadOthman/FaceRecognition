import cv2
import os
import pickle
import face_recognition
from datetime import datetime, timedelta
import time
import dlib
import numpy as np
import csv



def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1640,
        capture_height=1232,
        framerate=30,
        flip_method=0,
        display_width=820,
        display_height=616,
):
    return (
            "nvarguscamerasrc sensor-id=%d !"
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )

def append_attendance_record(person_id, detection_DT, loc_id):
    with open(attendance_csv_filename, "a", newline="") as a:
        writer1 = csv.writer(a)
        writer1.writerow([person_id, detection_DT, loc_id])

def read_attendance_csv():
    with open(attendance_csv_filename, 'r') as r:
        reader = csv.reader(r)
        next(reader)
        employee_attendance_data = [row for row in reader]

    return employee_attendance_data

def check_employee_previous_attendance(employee_data, empl_id, attendance_dt_str, loc_id):
    found = False
    for row in employee_data:
        if row[0] == empl_id and attendance_dt_str.split(" ")[0] in row[1] and row[2] == loc_id:
            found = True

    return found

def get_employee_latest_attendance_record(employee_data, empl_id):
    employee_att_list = []
    for row2 in employee_data:
        if row2[0] == empl_id:
            employee_att_list.append(row2)
    # Get the last row of the list (the latest record) then extract the DateTime at index 1
    return employee_att_list[-1][1]

def check_visitors_face_encodings(visitors_face_enc, face_encoding):
    visitor_encoding_match_bool = False
    for saved_encoding, saved_time in visitors_face_enc:
        visit_time_difference = datetime.now() - saved_time
        if visit_time_difference.total_seconds() < 300:
            if face_recognition.compare_faces([saved_encoding], face_encoding, tolerance=0.7)[0]:
                visitor_encoding_match_bool = True
                break

    return visitor_encoding_match_bool

def getDets(frame):
    '''
    Desc:
        Transform frame into gray scale and get face detection

    Args:
        frame
    Return:
        face detections and image in gray scale
    '''
    imgGRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    dets = detector(imgGRAY, 1)

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

def openAttendanceCSV():
    '''
    Desc:
        -open the .csv attendance file if exist
        -if not exist, create one and
    Return:
        the attendance csv path
    '''
    date_today = datetime.now().strftime("%Y%m%d")
    attendance_csv_filename = "attendance_" + date_today + ".csv"
    if not os.path.exists(attendance_csv_filename):
        with open(attendance_csv_filename, "w", newline="") as f1:
            writer = csv.writer(f1)
            writer.writerow(["EmployeeID", "DateTime", "LocationID"])
    return attendance_csv_filename

def getDataSet(folder_path='employee.p'):
    '''
    Args:
        folder_path: 'employee.p' as default

    Return:
         the encondings and name of known employee's images
    '''
    data_file = open(folder_path, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    knownFaceEncodings = data['encodings']
    knownNames = data['names']

    return knownFaceEncodings, knownNames

def getBestMatch(knownFaceEncodings, visitorEncoding, tolerance=0.6):
    '''
    Desc:
        determin the match and best match index according encoding distance difference
    Args:
        knownFaceEncodings, visitorEncoding, tolerance=0.6 as default
    Return:
         return matches and best match index
    '''
    match = face_recognition.compare_faces(knownFaceEncodings, visitorEncoding, tolerance=tolerance)
    faceDistance = face_recognition.face_distance(knownFaceEncodings, visitorEncoding)
    bestMatch = np.argmin(faceDistance)
    return match[bestMatch], bestMatch

def applyAttendIfRecord(record_found, employee_data, name, attendance_time, attendance_time_str, location_id):
    '''
    Desc:
        if previous record found today, then:
            -if the timed difference more than 5 min append new record
            -else, append new record
    Args:
        record_found, employee_data, name, attendance_time, attendance_time_str, location_id
    '''
    if record_found:
        #get the latest attendance
        employee_latest_att = get_employee_latest_attendance_record(employee_data, name)
        #get the time difference between now and latest attendance time
        time_difference = attendance_time - datetime.strptime(employee_latest_att, "%Y-%m-%d %H:%M:%S")
        if time_difference > timedelta(minutes=5):
            #if the time difference more than 5 min, record attendance
            print('Employee ' + name + ' detected at: ' + attendance_time_str)
            append_attendance_record(name, attendance_time_str, location_id)
        else:
            #these is the first comming of the employee
            append_attendance_record(name, attendance_time_str, location_id)

attendance_csv_filename=openAttendanceCSV()


#target faces
visitorFaceLocations=[]
visitorFaceEncodings = []
face_names = []
known_faces_detection_frame_count = 0
#unknow visitor
visitorsFaceEncodingsList = []
visitors_faces_detection_frame_count = 0
# Every second, 30 frames are captured and only 3 frames are processed
known_faces_frame_threshold = 2
visitors_faces_frame_threshold = 6
#load datasets and open the csv file
knownFaceEncodings, knownNames = getDataSet()
#print(knownNames, knownFaceEncodings)
location_id = '${LOCATION_ID}'
process_this_frame = True
cap = cv2.VideoCapture('Videos')

if __name__=="__main__":
    while True:
        success, frame = cap.read()
        if not success:
            break
        dets, imgGRAY = getDets(frame)
        for k, d in enumerate(dets):
            x, y, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            face_aligned = imgGRAY[y:y2, x:x2]
            face_normalized = processFace(face_aligned)
            #get face location and encodings
            visitorFaceLocations = face_recognition.face_locations(face_normalized, model="cnn", number_of_times_to_upsample=1)
            visitorFaceEncodings = face_recognition.face_encodings(face_normalized, visitorFaceLocations, model='large', num_jitters=1)
            for visitorEncoding, faceLocation in zip(visitorFaceEncodings, visitorFaceLocations):
                # Get current datetime as a formatted string
                attendance_time = datetime.now()
                #draw red rectangle for visitors
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                name = "unkown visitor"
                bestMatch, bestMatchIdx = getBestMatch(knownFaceEncodings, visitorEncoding)
                if bestMatch:
                    name = knownNames[bestMatchIdx]
                    print(name)
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    #cv2.putText(frame, name, (x-6, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.1, (0, 255, 0), 0.5)
                    #increment the known face
                    known_faces_detection_frame_count+=1
                    if known_faces_detection_frame_count == known_faces_frame_threshold:
                        #chack if the employee has already been detected
                        attendance_time_str = attendance_time.strftime("%Y-%m-%d %H:%M:%S")
                        employee_data = read_attendance_csv()
                        record_found = check_employee_previous_attendance(employee_data, name, attendance_time_str, location_id)
                        applyAttendIfRecord(record_found, employee_data, name, attendance_time, attendance_time_str, location_id)
                        known_faces_detection_frame_count=0
                else:
                    #these section for unkown visitors as an employee
                    print(name)
                    visitors_faces_detection_frame_count+=1
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    #cv2.putText(frame, name, (x + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
                    if visitors_faces_detection_frame_count == visitors_faces_frame_threshold:
                         #get match between the not employee visitor and unknown visitor encodings
                         visitorEncodingMatch = check_visitors_face_encodings(visitorsFaceEncodingsList, visitorFaceEncodings)
                         if visitorEncodingMatch:
                             print (visitorEncodingMatch)
                             continue
                         else:
                             visitorsFaceEncodingsList.append(visitorFaceEncodings)
                         visitors_faces_detection_frame_count = 0
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
