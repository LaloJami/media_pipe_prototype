# import necessary packages for hand gesture recognition project using Python OpenCV
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

import copy
import itertools

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
	max_num_hands=1, 
	min_detection_confidence=0.7,
	min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('asl_hand_gesture_v2')
# Load class names
f = open('gestureasl.names', 'r')
labels = f.read().split('\n')
f.close()
print(labels)




def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list



	

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)
while True:
	# Read each frame from the webcam
	_, frame = cap.read()
	#assert not isinstance(frame, type(None)), 'frame not found'
	if frame is not None:  # add this line

		x , y, c = frame.shape
	# Flip the frame vertically
	frame = cv2.flip(frame, 1)
	framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Get hand landmark prediction
	result = hands.process(framergb)
	
	className = ''
	# post process the result
	if result.multi_hand_landmarks:
		landmarks = []
		for handslms in result.multi_hand_landmarks:
			for lm in handslms.landmark:
				# print(id, lm)
				lmx = int(lm.x * x)
				lmy = int(lm.y * y)
				landmarks.append([lmx, lmy])
				# landmarks.append(pre_processed_landmark_list)
				# print(pre_processed_landmark_list)
				# print(len(pre_processed_landmark_list))

			# Drawing landmarks on frames
			mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
			# print((landmarks))
			# Bounding box calculation
			brect = calc_bounding_rect(framergb, handslms)
			# Landmark calculation
			landmark_list = calc_landmark_list(framergb, handslms)
			# Conversion to relative coordinates / normalized coordinates
			pre_processed_landmark_list = pre_process_landmark(
					landmark_list)
			# Predict gesture in Hand Gesture Recognition project
			prediction = model.predict([pre_processed_landmark_list])
			# print(prediction)
			classID = np.argmax(prediction)
			className = labels[classID].capitalize()

	# show the prediction on the frame
	#cv2.putText(frame, classNames, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
	cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

	# Show the final output
	cv2.imshow("Output", frame)
	if cv2.waitKey(1) == ord('q'):
		break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()