import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pyttsx3
import threading
import queue
import time
import os
import string

MODEL_PATH = 'sign_language_model.h5'

# Define Classes A-Z (excluding J and Z)
# They represent motion/gestures and are often excluded from static sign datasets
classes = list(string.ascii_uppercase)
classes.remove('J')
classes.remove('Z')

def build_cnn_model():
    """Builds and compiles the Keras CNN model for 24 classes."""
    print("Building new CNN model architecture...")
    model = Sequential([
        # Input: 64x64 Grayscale images
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output: 24 classes (Softmax for multi-class classification)
        Dense(24, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_or_create_model():
    """Loads a trained model if it exists, otherwise generates an untrained one."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        return load_model(MODEL_PATH)
    else:
        print("Model file not found. Creating a baseline untrained model.")
        model = build_cnn_model()
        model.save(MODEL_PATH)
        print("*" * 60)
        print("WARNING: The model is currently untrained. Predictions will be random!")
        print("You must train this model using a dataset (e.g., Sign Language MNIST) for accuracy.")
        print("*" * 60)
        return model

# Text-to-Speech Worker Queue
speech_queue = queue.Queue()

def tts_worker():
    """Worker thread to handle text-to-speech asynchronously."""
    try:
        engine = pyttsx3.init()
        # You can adjust speech rate here if desired -> engine.setProperty('rate', 150)
        while True:
            text = speech_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()
    except Exception as e:
        print(f"TTS Engine Error: {e}")

def speak(text):
    speech_queue.put(text)

def main():
    # 1. Initialize CNN Model
    model = load_or_create_model()
    
    # 2. Start TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # 3. MediaPipe setup for Hand Detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # min_detection_confidence is set a bit higher for better hand targeting
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    last_spoken = {}
    speak_cooldown = 3.0 # Wait 3 seconds before repeating the same letter output

    print("\nStarting sign language recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip frame horizontally for a more natural mirror/selfie experience
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # MediaPipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Initialize bounding box coordinates limits
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                # Iterate through 21 landmarks to find the bounding box
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y

                # Pad the bounding box slightly
                padding = 20
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)

                # Draw green bounding box surrounding the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw MediaPipe structural landmarks over the fingers
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Validate ROI boundaries
                if y_max > y_min and x_max > x_min:
                    # Extract the Region of Interest (ROI)
                    roi = frame[y_min:y_max, x_min:x_max]

                    # PREPROCESSING REQUIREMENTS: Grayscale, resize to 64x64
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (64, 64))
                    
                    # Normalize pixel values [0,1] and reshape to (BATCH, X, Y, CHANNELS) for Keras
                    roi_normalized = roi_resized / 255.0
                    roi_reshaped = np.reshape(roi_normalized, (1, 64, 64, 1))

                    # 4. Predict the class Using Keras CNN (Using model() is faster in a loop than .predict())
                    prediction_tensor = model(roi_reshaped, training=False)
                    prediction_array = prediction_tensor.numpy()[0]
                    
                    predicted_class_idx = np.argmax(prediction_array)
                    confidence = float(prediction_array[predicted_class_idx])
                    
                    predicted_char = classes[predicted_class_idx]

                    # 5. Display Prediction and Confidence
                    label = f'{predicted_char} ({confidence:.2f})'
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # 6. Auditory Output (Text-To-Speech) via Pyttsx3
                    current_time = time.time()
                    # A threshold keeps random noise from speaking, though an untrained model will misfire. 
                    if confidence > 0.4:  
                        if predicted_char not in last_spoken or (current_time - last_spoken[predicted_char] > speak_cooldown):
                            speak(predicted_char)
                            last_spoken[predicted_char] = current_time

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Shutdown sequence
    cap.release()
    cv2.destroyAllWindows()
    
    # Gracefully stop the TTS worker thread
    speech_queue.put(None)
    tts_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
