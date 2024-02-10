import cv2
import mediapipe as mp
import numpy as np
import pickle

def normalize_landmarks(hand_landmarks, frame_shape):
    x_, y_ = [], []
    data_aux = []

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y

        x_.append(x)
        y_.append(y)

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))

    return data_aux, x_, y_

def draw_prediction(frame, x1, y1, x2, y2, predicted_character):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                cv2.LINE_AA)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {0: 'I LOVE YOU', 1: 'FUCK', 2: 'BODOH', 3: 'TEMAN', 4: 'HAI'}

# Streamlit app
st.title("Sign Language Detector")

# OpenCV video capture
cap = cv2.VideoCapture(0)

# MediaPipe hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Main Streamlit app loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #Uncomment this block if you want to draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

            # Process hand landmarks for prediction
            data_aux, normalized_x, normalized_y = normalize_landmarks(hand_landmarks, frame.shape)

            x1 = int(min(normalized_x) * W) - 10
            y1 = int(min(normalized_y) * H) - 10
            x2 = int(max(normalized_x) * W) - 10
            y2 = int(max(normalized_y) * H) - 10

            # Assuming the model expects 84 features
            expected_features = 84

            # Ensure data_aux has the correct number of features
            if len(data_aux) < expected_features:
                # Pad with zeros if it has fewer features
                data_aux.extend([0.0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                # Truncate if it has more features
                data_aux = data_aux[:expected_features]
            
            # Convert to NumPy array for prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_value = int(prediction[0])

            if predicted_value in labels_dict:
                predicted_character = labels_dict[predicted_value]
            else:
                print(f"Warning: Predicted value {predicted_value} is not in the labels_dict.")
                predicted_character = "UNKNOWN"

            draw_prediction(frame, x1, y1, x2, y2, predicted_character)

    # Display the frame in Streamlit
    st.image(frame, channels="BGR")

# Release the camera and close the Streamlit app
cap.release()
cv2.destroyAllWindows()
