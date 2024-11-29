import os
import cv2
import pickle
import numpy as np
from skimage.transform import resize
# Load the trained SVM model
with open('./SVM_Model.p', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load saved parking positions from the pickle file
try:
    with open('positionList', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Define the width and height of parking slots
width, height = 48, 21

def mouseClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open('positionList', 'wb') as f:
        pickle.dump(posList, f)

def extract_features(image):
    """
    Extracts features from an image for the SVM model.
    The image is resized to match the dimensions used during training and flattened.
    """
    resized_img = resize(image, (25, 27))  # Adjust size if needed
    features = resized_img.flatten()
    return features

# Initialize video capture
# path = os.path.join('.','Use_The_Model', 'parking_1920_1080.mp4')
video_path = "./data/parking_1920_1080_loop.mp4"
cap = cv2.VideoCapture(video_path)

frame_skip = 3  # Process every 3rd frame for faster performance
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break



    # Convert the frame to grayscale for feature extraction
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    empty_count = 0  # Initialize counter for empty slots

    for pos in posList:
        x, y = pos

        # Crop the grayscale frame around the parking slot
        crop_img = grayscale_frame[y:y + height, x:x + width]

        # Extract features from the cropped image
        features = extract_features(crop_img)
        features = np.array([features])  # Reshape to match input format for the model

        # Predict using the SVM model
        prediction = svm_model.predict(features)[0]

        # Assign color based on model prediction
        if prediction == 0:  # 0 represents 'empty'
            color = (0, 255, 0)  # Green for empty
            empty_count += 1  # Increment count if slot is empty
        else:  # 1 represents 'not_empty'
            color = (0, 0, 255)  # Red for not empty

        # Draw rectangle around the parking slot on the image
        cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, 2)

    # Display the number of empty slots on the frame
    cv2.putText(frame, f'Empty Slots: {empty_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame with marked parking slots
    cv2.imshow('Parking Slot Detection', frame)
    cv2.setMouseCallback('Parking Slot Detection', mouseClick)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()