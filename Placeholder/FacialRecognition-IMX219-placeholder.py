import csv
import time
from datetime import datetime, timedelta
import os
import pickle

import face_recognition
import cv2
import numpy as np


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


# Load the employee data from the encodings.pickle file

#with open("\Face Recognition\FaceRecognition\globaldws_employees_faces.p", 'rb') as file:
#    employee_data = pickle.load(file)


# Get the employee face encodings and IDs
#known_face_encodings = employee_data['encodings']
#known_face_ids = employee_data['names']

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_faces_detection_frame_count = 0

visitors_face_encodings = []
visitors_faces_detection_frame_count = 0
location_id = '${LOCATION_ID}'
process_this_frame = True
last_display_time = time.time()

counter = 0  # for skipping frames

# Every second, 30 frames are captured and only 3 frames are processed
known_faces_frame_threshold = 2
visitors_faces_frame_threshold = 6

# Define the name of the CSV file based on the current date
date_today = datetime.now().strftime("%Y%m%d")
attendance_csv_filename = "attendance_" + date_today + ".csv"

# Check if the file exists; if the file does not exist, create the file and write the header
if not os.path.exists(attendance_csv_filename):
    with open(attendance_csv_filename, "w", newline="") as f1:
        writer = csv.writer(f1)
        writer.writerow(["EmployeeID", "DateTime", "LocationID"])


if __name__ == '__main__':
    # Get a reference to the CSI camera (as defined by the gstreamer pipeline above)
    #video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        # Only process every 10th frame of video to save time
        counter += 1
        if counter % 10 == 0:
            process_this_frame = True
        else:
            process_this_frame = False

        if process_this_frame:

            #put Guassian filter to reduce noise
            #image = cv2.GaussianBlur(frame, (5, 5), 0)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video; using 'CNN' model for GPU utilization
            face_locations = face_recognition.face_locations(imageRGB, model='cnn')
            face_encodings = face_recognition.face_encodings(imageRGB, face_locations)

            if face_locations:
                #Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            #face_names = []
            #for face_encoding in face_encodings:
                # Get current datetime as a formatted string
                #attendance_time = datetime.now()
                #attendance_time_str = attendance_time.strftime("%Y-%m-%d %H:%M:%S")

                # See if the face is a match for the known face(s)
                #matches = face_recognition.compare_faces(
                #    known_face_encodings, face_encoding, tolerance=0.4)  # Lowered tolerance from 0.6 (default)
                # to 0.4 for a more strict and accurate recognition.
                #name = "Unknown Visitor"

                # Use the known face with the smallest distance to the new face
                #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                #best_match_index = np.argmin(face_distances)

                # If a match is found in the pretrained face encodings:
                #if matches[best_match_index]:
                    #name = known_face_ids[best_match_index]

                    #Display the results
                #for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Draw a box around the face
                    #cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    #cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                    # Increment the detection frame count
                    #known_faces_detection_frame_count += 1


                    #if known_faces_detection_frame_count == known_faces_frame_threshold:
                     #   employee_id = name

                        # Check if the employee has already been detected in the same day
                      #  employee_data = read_attendance_csv()
                       # record_found = check_employee_previous_attendance(
                        #    employee_data, employee_id, attendance_time_str, location_id)

                        #if record_found:
                            # Find the latest DateTime for this employee's attendance
                            #employee_latest_att = get_employee_latest_attendance_record(employee_data, employee_id)
                            #time_difference = attendance_time - datetime.strptime(employee_latest_att, "%Y-%m-%d %H:%M:%S")
                            #if time_difference > timedelta(minutes=5):
                            #    print('Employee ' + employee_id + ' detected at: ' + attendance_time_str)
                            #    append_attendance_record(employee_id, attendance_time_str, location_id)

                        #if not record_found:
                        #   append_attendance_record(employee_id, attendance_time_str, location_id)

                        # Reset the detection time
                        #known_faces_detection_frame_count = 0

                # If no match was found in the pre-trained face encodings:
                #else:
                #    visitors_faces_detection_frame_count += 1
                #    if visitors_faces_detection_frame_count == visitors_faces_frame_threshold:
                #        # Check if the encoding matches any that have been saved within the last 5 minutes
                #        visitor_encoding_match = check_visitors_face_encodings(visitors_face_encodings, face_encoding)

                #        # If the unknown face has not been previously detected, save the encoding and time of detection then append a record to the attendance CSV file
                #        if not visitor_encoding_match:
                #            visitors_face_encodings.append((face_encoding, datetime.now()))
                #            print("NEW VISITOR DETECTED AT:", attendance_time_str)
                #            # Append visitor record containing a sentinel value (99) for Unknown Visitors as ID
                #            append_attendance_record(str(99), attendance_time_str, location_id)
                #        else:
                #            # Skip the detection and move on to the next frame
                #            continue

                #        # Reset the detection time
                #        visitors_faces_detection_frame_count = 0

                #face_names.append(name)


            # # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release handle to the webcam
    #video_capture.release()
    #cv2.destroyAllWindows()
