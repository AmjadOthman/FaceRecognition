# facial-recognition | Occupancy Management

This README will guide users to set up the Jetson Nano to run the `facial-recognition` solution by GlobalDWS. 

This solution utilizes the `face_recognition` Python library to train a face recognition model first and after creating this data file, we can run the facial recognition solution to identify whether the detected face is an employee or an unknown visitor. When an employee is detected, a monitored CSV file will be appended with this employee's attendance record. If an unknown visitor is detected, the CSV file will be appended with a sentinel value (99) as an ID alongside other information.

The CSV file is monitored by another Python script named `SendAttendanceTelemetry.py`. This script waits for a change to occur in the CSV file and sends the latest record as telemetries to an IoT Central table.

---

[TOC]

---

## Setup Steps

### Python 3.8

**Python3.8** is needed to run the script due to some package compatibility issues with older Python versions (mainly Azure IoT-related packages). If Python3.8 is not installed already, run the following commands:

```shell
sudo apt update && sudo apt install software-properties-common && sudo add-apt-repository ppa:deadsnakes/ppa
```

ENTER

```shell
sudo apt install python3.8 python3.8-dev python3.8-venv python3.8-distutils
python3.8 -m ensurepip
```

---

### Clone the facial-recognition GitHub repository

To clone the GitHub repository that is on GlobalDWS's GitHub projects page, we must first install the `git` command on the Jetson Nano. After doing so, we must clone the repo to the home directory `~`. To do so, run the following commands:

```shell
sudo apt install git
cd ~ && git clone https://github.com/globaldws/facial-recognition.git
```

After doing so, we must use `envsubst` to change the environment variable placeholders in the Python placeholder scripts. The locationID will be attached alongside other data in telemetries, so choose a unique value for each location where this solution is set up. To do so, run the following command:

```shell
cd facial-recognition
export LOCATION_ID= <a value of your choice>
envsubst < Placeholder/FacialRecognition-IMX219-placeholder.py > FacialRecognition-IMX219.py
envsubst < Placeholder/SendAttendanceTelemetry-placeholder.py > SendAttendanceTelemetry.py
```

Then, remove the `Placeholder` directory recursively by using the following command:

```shell
rm -rf Placeholder/
```

---

### Create a Python virtual environment and install requirements

We should create a Python venv called `.venv` inside of the `facial-recognition` directory and install all requirements in the `requirements.txt` file. To do so, run the following commands:

```shell
cd ~/facial-recognition/ && python3.8 -m venv .venv
source .venv/bin/activate && pip install wheel && pip install -r requirements.txt
# Copy the OpenCV you built from source into the venv (for GStreamer capabilities)
cp -r /usr/local/lib/python3.8/dist-packages/cv2/ ~/facial-recognition/.venv/lib/python3.8/site-packages/
```

----

### Train model

To train a facial recognition model (it isn't a model, per se, it's actually a dataset containing all face encodings for each person), a script named `train_model.py` will process images that you provide it in the following format:

```
.
└── dataset
    ├── 01
    │   ├── image_1.jpeg
    │   └── ...
    ├── 02
    │   ├── image_1.jpeg
    │   └── ...
    ├── 03
    │   ├── image_1.jpeg
    │   └── ...
    └── ...
```

Create a folder named `dataset` and then create subdirectories with the names/IDs of the employees. In each subdirectory, provide a set of images (~20) where a person's face is clearly visible and each image is taken from a slightly different angle. The name of the images does not matter, as there is no regex used to look for specific filename patterns.

After this script processes all the data, it dumps (saves) the face encodings and names/IDs into a file named `globaldws_employees_faces.dat`, which will be used in the facial recognition Python script.

---

### Enable systemd services and timers

A systemd service is needed so that the script runs automatically at specific times defined in the respective systemd timers. A systemd timer is needed so that we can run the respective systemd services at predefined times as needed. A thorough explanation of systemd timers is available on [this link](https://wiki.archlinux.org/title/systemd/Timers).

Five services and 3 timers are provided in the project. I will explain each group of services+timer under one heading. We are going to move the timers directly to `/etc/systemd/system/`, but we have to substitute the environment variable placeholders inside the service files using `envsubst`.

#### FacialRecognition

This group contains the following:

-  `FacialRecognition.service` : runs the `FacialRecognition-IMX219.py` Python script at boot up
- `FacialRecognitionRestarter` `.service` and `.timer` : restarts the `FacialRecognition.service` 1 minute after midnight

#### SendAttendanceTelemetry

This group contains the following:

- `SendAttendanceTelemetry.service` : runs the `SendAttendanceTelemetry.py` Python script at boot up
- `SendAttendanceTelemetryRestarter` `.service` and `.timer` : restarts the `SendAttendanceTelemetry.service` 30 seconds after midnight

#### CSVFileRemover

This group contains a service and a timer with the same name; they remove CSV files from `facial-recognition/` directory at exactly midnight.

We need to perform `envsubst` on the 3 services placeholder files and move the `restarter` service and timer files to `/etc/systemd/system/`. To do so, run the following commands:

```shell
cd ~/facial-recognition/ServicesAndTimersPlaceholder/
# perform envsubst
sudo bash -c "envsubst < FacialRecognition-placeholder.service > /etc/systemd/system/FacialRecognition.service"
sudo bash -c "envsubst < SendAttendanceTelemetry-placeholder.service > /etc/systemd/system/SendAttendanceTelemetry.service"
sudo bash -c "envsubst < CSVFileRemover-placeholder.service > /etc/systemd/system/CSVFileRemover.service"
# move services and timers
sudo mv FacialRecognitionRestarter.service /etc/systemd/system/FacialRecognitionRestarter.service
sudo mv FacialRecognitionRestarter.timer /etc/systemd/system/FacialRecognitionRestarter.timer
sudo mv SendAttendanceTelemetryRestarter.service /etc/systemd/system/SendAttendanceTelemetryRestarter.service
sudo mv SendAttendanceTelemetryRestarter.timer /etc/systemd/system/SendAttendanceTelemetryRestarter.timer
sudo mv CSVFileRemover.timer /etc/systemd/system/CSVFileRemover.timer
```

After all services and timers are in the `/etc/systemd/system/` directory, remove the `ServicesAndTimersPlaceholder` directory recursively by running the following command:

```shell
rm -rf ~/facial-recognition/ServicesAndTimersPlaceholder/
```

To enable the services and timers, run the following commands:

```shell
sudo systemctl enable FacialRecognition.service
sudo systemctl enable FacialRecognitionRestarter.service
sudo systemctl enable FacialRecognitionRestarter.timer
sudo systemctl enable SendAttendanceTelemetry.service
sudo systemctl enable SendAttendanceTelemetryRestarter.service
sudo systemctl enable SendAttendanceTelemetryRestarter.timer
sudo systemctl enable CSVFileRemover.service
sudo systemctl enable CSVFileRemover.timer
```

To start a service now, run the following command:

```shell
sudo systemctl start <SERVICE-NAME>
```

To check the status of a service, run the following command:

```shell
sudo systemctl status <SERVICE-NAME>
```

To stop a service now, run the following command:

```shell
sudo systemctl stop <SERVICE-NAME>
```

To disable a service, run the following command:

```shell
sudo systemctl disable <SERVICE-NAME>
```

To check the logs of a service, run the following command:

```shell
sudo journalctl -u <SERVICE-NAME>
```

---

# End of README