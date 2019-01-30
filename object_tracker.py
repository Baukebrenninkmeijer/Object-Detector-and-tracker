import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from mrcnn import visualize
import random
import time
# from twilio.rest import Client


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Twilio config
# twilio_account_sid = 'YOUR_TWILIO_SID'
# twilio_auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
# twilio_phone_number = 'YOUR_TWILIO_SOURCE_PHONE_NUMBER'
# destination_phone_number = 'THE_PHONE_NUMBER_TO_TEXT'
# client = Client(twilio_account_sid, twilio_auth_token)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
parked_car_boxes = None

VIDEO_SOURCE = "images/duif.mp4"

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# How many frames of video we've seen in a row with a parking space open
free_space_frames = 0

# Have we sent an SMS alert yet?
sms_sent = False

COLORS = visualize.random_colors(10)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

frame_start = time.time()
prev = None
overlaps = None
# Loop over each frame of video
while video_capture.isOpened():
    frame_end = time.time()
    print(f'frame time: {frame_end-frame_start}')
    frame_start = time.time()
    success, frame = video_capture.read()
    if not success:
        break

    height, width, depth = frame.shape
    if height > 720:
        SCALE = (720/height)
        frame = cv2.resize(frame,(int(width*SCALE),int(height*SCALE)))

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Run the image through the Mask R-CNN model to get results.
    start = time.time()
    results = model.detect([rgb_image], verbose=0)
    end = time.time()

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)
    roi_start = time.time()
    clone = frame.copy()
    used_colors = []
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(clone, f"duration: {end-start:0.3}s", (10, 20), font, 0.8, (0, 0, 0))
    if prev is None:
        prev = (r['rois'], r['class_ids'], visualize.random_colors(r['rois'].shape[0]))

    prev_rois, prev_class_ids, prev_colors = prev
    overlaps = mrcnn.utils.compute_overlaps(r['rois'], prev_rois)
    for roi, mask, class_id, score, overlap in zip(r['rois'], np.moveaxis(r['masks'], 2, 0), r['class_ids'], r['scores'], overlaps):
        color = random.choice(COLORS)

        if overlap.size != 0:
            index_of_overlap = np.argmax(overlap)
            if np.max(overlap) > 0.5 and class_id == prev_class_ids[index_of_overlap]:
                color = prev_colors[index_of_overlap]

        used_colors.append(color)
        clone = visualize.apply_mask(clone, mask, color)
        y1, x1, y2, x2 = roi
        print(f'({x1},{y1})')
        print(f'({x2},{y2})')
        cv2.rectangle(clone, (x1, y1), (x2, y2), thickness=2, color=[i * 255 for i in color])

        label = class_names[class_id]
        cv2.putText(clone, f"{label}:{score:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))


        # if parked_car_boxes is None:
        #     # This is the first frame of video - assume all the cars detected are in parking spaces.
        #     # Save the location of each car as a parking space box and go to the next frame of video.
        #     parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # else:
        # We already know where the parking spaces are. Check if any are currently unoccupied.

        # Get where cars are currently located in the frame
        # print(parked_car_boxes)
        # car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # print(car_boxes)
        # # See how much those cars overlap with the known parking spaces
        # overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        # Assume no spaces are free until we find one that is free
        # free_space = False

        # Loop through each known parking space box
        # for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
        #
        #     # For this parking space, find the max amount it was covered by any
        #     # car that was detected in our image (doesn't really matter which car)
        #     max_IoU_overlap = np.max(overlap_areas)
        #
        #     # Get the top-left and bottom-right coordinates of the parking area
        #     y1, x1, y2, x2 = parking_area
        #
        #     # Check if the parking space is occupied by seeing if any car overlaps
        #     # it by more than 0.15 using IoU
        #     if max_IoU_overlap < 0.15:
        #         # Parking space not occupied! Draw a green box around it
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #         # Flag that we have seen at least one open space
        #         free_space = True
        #     else:
        #         # Parking space is still occupied - draw a red box around it
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #
        #     # Write the IoU measurement inside the box
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

        # If at least one space was free, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.
        # if free_space:
        #     free_space_frames += 1
        # else:
        #     # If no spots are free, reset the count
        #     free_space_frames = 0

        # If a space has been free for several frames, we are pretty sure it is really free!
        # if free_space_frames > 10:
        #     # Write SPACE AVAILABLE!! at the top of the screen
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

            # If we haven't sent an SMS yet, sent it!
            # if not sms_sent:
            #     print("SENDING SMS!!!")
            #     message = client.messages.create(
            #         body="Parking space open - go go go!",
            #         from_=twilio_phone_number,
            #         to=destination_phone_number
            #     )
            #     sms_sent = True

        # Show the frame of video on the screen
    prev = (r['rois'], r['class_ids'], used_colors)
    cv2.imshow('Video', clone)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
