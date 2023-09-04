import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import paho.mqtt.client as mqtt
import requests
import werkzeug
import yaml
from PIL import Image, ImageOps
from prometheus_client import Counter
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

from queries import get_common_name
from webui import app

classifier = None
config = None
firstmessage = True

DBPATH = './data/speciesid.db'
CATEGORY_BACKGROUND = 964  # Model classification category for background

BIRD_COUNTER = Counter("detected", "Counter of detected species.", ("species", ))


def classify(image):
    tensor_image = vision.TensorImage.create_from_array(image)
    categories = classifier.classify(tensor_image)
    return categories.classifications[0].categories


def on_connect(client, userdata, flags, rc):
    logging.info("MQTT Connected")

    # we are going subscribe to frigate/events and look for bird detections there
    client.subscribe(config['frigate']['main_topic'] + "/events")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        logging.error("Unexpected disconnection, trying to reconnect")
        while True:
            try:
                client.reconnect()
                break
            except Exception as e:
                logging.error(f"Reconnection failed due to {e}, retrying in 60 seconds")
                time.sleep(60)
    else:
        logging.error("Expected disconnection")


def set_sublabel(frigate_url, frigate_event, sublabel):
    post_url = frigate_url + "/api/events/" + frigate_event + "/sub_label"

    # frigate limits sublabels to 20 characters currently
    if len(sublabel) > 20:
        sublabel = sublabel[:20]

        # Create the JSON payload
    payload = {
        "subLabel": sublabel
    }

    # Set the headers for the request
    headers = {
        "Content-Type": "application/json"
    }

    # Submit the POST request with the JSON payload
    response = requests.post(post_url, data=json.dumps(payload), headers=headers)

    # Check for a successful response
    if response.status_code == 200:
        logging.debug("Sublabel set successfully to: " + sublabel)
    else:
        logging.debug("Failed to set sublabel. Status code:", response.status_code)


def on_message(client, userdata, message):
    conn = sqlite3.connect(DBPATH)

    # Convert the MQTT payload to a Python dictionary
    payload_dict = json.loads(message.payload)

    # Extract the 'after' element data and store it in a dictionary
    after_data = payload_dict.get('after', {})

    if (after_data['camera'] in config['frigate']['camera'] and
            after_data['label'] == config['frigate']['object']):

        frigate_event = after_data['id']
        frigate_url = config['frigate']['frigate_url']
        snapshot_url = frigate_url + "/api/events/" + frigate_event + "/snapshot.jpg"

        logging.debug("Getting image for event: " + frigate_event)
        logging.debug("Here's the URL: " + snapshot_url)
        # Send a GET request to the snapshot_url
        params = {
            "crop": 1,
            "quality": 95
        }
        response = requests.get(snapshot_url, params=params)
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Open the image from the response content and convert it to a NumPy array
            image = Image.open(BytesIO(response.content))

            file_path = "fullsized.jpg"  # Change this to your desired file path
            image.save(file_path, format="JPEG")  # You can change the format if needed

            # Resize the image while maintaining its aspect ratio
            max_size = (224, 224)
            image.thumbnail(max_size)

            # Pad the image to fill the remaining space
            padded_image = ImageOps.expand(image, border=((max_size[0] - image.size[0]) // 2,
                                                          (max_size[1] - image.size[1]) // 2),
                                           fill='black')  # Change the fill color if necessary

            file_path = "shrunk.jpg"  # Change this to your desired file path
            padded_image.save(file_path, format="JPEG")  # You can change the format if needed

            np_arr = np.array(padded_image)

            categories = classify(np_arr)
            category = categories[0]
            index = category.index
            score = category.score
            display_name = category.display_name
            category_name = category.category_name

            start_time = datetime.fromtimestamp(after_data['start_time'])
            formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info("%s: %s", formatted_start_time, str(category))

            if index != CATEGORY_BACKGROUND and score > config['classification']['threshold']:
                cursor = conn.cursor()

                # Check if a record with the given frigate_event exists
                cursor.execute("SELECT * FROM detections WHERE frigate_event = ?", (frigate_event,))
                result = cursor.fetchone()

                if result is None:
                    # Insert a new record if it doesn't exist
                    logging.info("No record yet for this event. Storing.")
                    cursor.execute("""  
                        INSERT INTO detections (detection_time, detection_index, score,  
                        display_name, category_name, frigate_event, camera_name) VALUES (?, ?, ?, ?, ?, ?, ?)  
                        """, (formatted_start_time, index, score, display_name, category_name, frigate_event, after_data['camera']))
                    BIRD_COUNTER.labels(display_name).inc()
                    # set the sublabel
                    set_sublabel(frigate_url, frigate_event, get_common_name(display_name))
                else:
                    logging.debug("There is already a record for this event. Checking score")
                    # Update the existing record if the new score is higher
                    existing_score = result[3]
                    if score > existing_score:
                        logging.info("New score is higher. Updating record with higher score.")
                        cursor.execute("""  
                            UPDATE detections  
                            SET detection_time = ?, detection_index = ?, score = ?, display_name = ?, category_name = ?  
                            WHERE frigate_event = ?  
                            """, (formatted_start_time, index, score, display_name, category_name, frigate_event))
                        BIRD_COUNTER.labels(display_name).inc()
                        # set the sublabel
                        set_sublabel(frigate_url, frigate_event, get_common_name(display_name))
                    else:
                        logging.debug("New score is lower.")

                # Commit the changes
                conn.commit()


        else:
            logging.error(f"Error: Could not retrieve the image. Status code: {response.status_code}")

    conn.close()


def setupdb():
    conn = sqlite3.connect(DBPATH)
    cursor = conn.cursor()
    cursor.execute("""    
        CREATE TABLE IF NOT EXISTS detections (    
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            detection_time TIMESTAMP NOT NULL,  
            detection_index INTEGER NOT NULL,  
            score REAL NOT NULL,  
            display_name TEXT NOT NULL,  
            category_name TEXT NOT NULL,  
            frigate_event TEXT NOT NULL UNIQUE,
            camera_name TEXT NOT NULL 
        )    
    """)
    conn.commit()

    conn.close()

def load_config():
    global config
    file_path = './config/config.yml'
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    log_level = config.get("logging", {}).get("level", "DEBUG")
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s {%(filename)s:%(lineno)s|%(name)s}",
        level=levels.get(log_level, logging.DEBUG)
    )

def run_webui():
    logging.info("Starting flask app")
    app.run(debug=False, host=config['webui']['host'], port=config['webui']['port'])


def run_mqtt_client():
    logging.info("Starting MQTT client. Connecting to: " + config['frigate']['mqtt_server'])
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    client = mqtt.Client("birdspeciesid" + current_time)
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.on_connect = on_connect
    # check if we are using authentication and set username/password if so
    if config['frigate']['mqtt_auth']:
        username = config['frigate']['mqtt_username']
        password = config['frigate']['mqtt_password']
        client.username_pw_set(username, password)

    client.connect(config['frigate']['mqtt_server'])
    client.loop_start()
    return client


def main():
    load_config()

    # Initialize the image classification model
    base_options = core.BaseOptions(file_name=config['classification']['model'], use_coral=False, num_threads=4)

    # Enable Coral by this setting
    classification_options = processor.ClassificationOptions(max_results=1, score_threshold=0)
    options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)

    # create classifier
    global classifier
    classifier = vision.ImageClassifier.create_from_options(options)

    # setup database
    setupdb()
    logging.debug("Starting threads for Flask and MQTT")

    mqtt = run_mqtt_client()

    try:
        run_webui()
    except KeyboardInterrupt:
        pass

    mqtt.loop_stop()
    logging.info("All done, bye.")


if __name__ == '__main__':
    main()
