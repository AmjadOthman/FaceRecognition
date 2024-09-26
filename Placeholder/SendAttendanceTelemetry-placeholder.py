import asyncio
import csv
import json
import logging
import time

import pandas as pd
from datetime import datetime
from pathlib import Path

from azure.iot.device import Message
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device.aio import ProvisioningDeviceClient

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


async def provision_device(provisioning_host, id_scope, registration_id, symmetric_key, model_id):
    provisioning_device_client = ProvisioningDeviceClient.create_from_symmetric_key(
        provisioning_host=provisioning_host,
        registration_id=registration_id,
        id_scope=id_scope,
        symmetric_key=symmetric_key,
    )
    provisioning_device_client.provisioning_payload = {"modelId": model_id}
    return await provisioning_device_client.register()


async def send_telemetry_from_nano(device_client, telemetry_msg):
    msg = Message(json.dumps(telemetry_msg, default=str))
    msg.content_encoding = "utf-8"
    msg.content_type = "application/json"
    print("Sent message")
    await device_client.send_message(msg)


async def sendAttendance():
    # ––––– Define IOT central Variables saved in the CSV file ––––– #
    env_var_path = '${HOME}/facial-recognition/DeviceEnvironment_Attendance.csv'
    with open(env_var_path, newline='') as fp:
        csvreader = csv.DictReader(fp)
        for row in csvreader:
            Device = row

    IOTHUB_DEVICE_SECURITY_TYPE = Device['IOTHUB_DEVICE_SECURITY_TYPE']
    IOTHUB_DEVICE_DPS_ID_SCOPE = Device['IOTHUB_DEVICE_DPS_ID_SCOPE']
    IOTHUB_DEVICE_DPS_DEVICE_KEY = Device['IOTHUB_DEVICE_DPS_DEVICE_KEY']
    IOTHUB_DEVICE_DPS_DEVICE_ID = Device['IOTHUB_DEVICE_DPS_DEVICE_ID']
    IOTHUB_DEVICE_DPS_ENDPOINT = Device['IOTHUB_DEVICE_DPS_ENDPOINT']
    model_id = Device['model_id']

    # ––––– Connecting to IoT Central ––––– #
    print("-" * 30)

    switch = IOTHUB_DEVICE_SECURITY_TYPE
    if switch == "DPS":
        provisioning_host = (
            IOTHUB_DEVICE_DPS_ENDPOINT
            if IOTHUB_DEVICE_DPS_ENDPOINT
            else "global.azure-devices-provisioning.net"
        )

        id_scope = IOTHUB_DEVICE_DPS_ID_SCOPE
        registration_id = IOTHUB_DEVICE_DPS_DEVICE_ID
        symmetric_key = IOTHUB_DEVICE_DPS_DEVICE_KEY

        registration_result = await provision_device(
            provisioning_host, id_scope, registration_id, symmetric_key, model_id)

        if registration_result.status == "assigned":
            print("Device was assigned")
            print(registration_result.registration_state.assigned_hub)
            print(registration_result.registration_state.device_id)

            device_client = IoTHubDeviceClient.create_from_symmetric_key(
                symmetric_key=symmetric_key,
                hostname=registration_result.registration_state.assigned_hub,
                device_id=registration_result.registration_state.device_id,
                product_info=model_id,
            )
        else:
            raise RuntimeError("Could not provision device. Aborting Plug and Play device connection.")

    else:
        raise RuntimeError(
            "At least one choice needs to be made for complete functioning of this sample."
        )

    await device_client.connect()
    print('IoTC device client connected')
    # ––––– End of Connecting to IoT Central ––––– #

    dt = datetime.now()
    date_ymd = dt.strftime("%Y%m%d")  # e.g. 20230209

    csv_filename = 'attendance_' + date_ymd + '.csv'
    csv_filepath = Path('${HOME}/facial-recognition/' + csv_filename)

    try:
        latest_attendance_data = pd.read_csv(csv_filepath)
        last_locationID = latest_attendance_data['LocationID'].iloc[-1]
        last_employeeID = latest_attendance_data['EmployeeID'].iloc[-1]
        last_attendance_dt = latest_attendance_data['DateTime'].iloc[-1]

        await send_telemetry_from_nano(device_client, {
            "locationID": int(last_locationID),
            "EmpId": int(last_employeeID),
            "DateTimeStamp": str(last_attendance_dt),
        })
        logging.info('Successfully sent attendance telemetry')

    except:
        logging.error("Could not read values from CSV file. Aborting attendance telemetry upload.")
    finally:
        await device_client.shutdown()
        print('IoTC device client disconnected')
        print("-" * 30)

    await asyncio.sleep(5)
    print("Async function completed")


# Define the event handler
class MyEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Make sure that the event came from a modified file (not directory)
        if event.is_directory:
            return
        # Print and send attendace telemetry
        print(f"{event.src_path} has been modified at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        asyncio.run(sendAttendance())


if __name__ == '__main__':
    # To check if `attendance_<DATE>.csv` file exists
    date = datetime.now().strftime("%Y%m%d")  # e.g. 20230209
    csv_filepath = Path('${HOME}/facial-recognition/' + 'attendance_' + date + '.csv')

    # If it doesn't exist, create the file with the relevant headers
    if not csv_filepath.exists():
        with open(csv_filepath, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["EmployeeID", "DateTime", "LocationID"])
        if csv_filepath.exists():
            print('File created at {}'.format(csv_filepath))
        time.sleep(2)

    # Run the asynchronous function in a separate thread
    observer = Observer()
    event_handler = MyEventHandler()
    observer.schedule(event_handler, str(csv_filepath), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
