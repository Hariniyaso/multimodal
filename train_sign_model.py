import os
import glob
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import kagglehub
import cv2
import sys
import mediapipe as mp
from sklearn.model_selection import train_test_split

def build_dnn_model():
    """Generates a highly robust Dense Neural Network for MediaPipe landmarks."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(42,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_landmarks(image_path, hands):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # We compute relative coordinates normalized to the wrist to make it invariant to location/scale
        wrist = hand_landmarks.landmark[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append(lm.x - wrist.x)
            coords.append(lm.y - wrist.y)
            
        coords = np.array(coords)
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val
            
        return coords
    return None

def main():
    print("-" * 60)
    print("Initiating Enhanced MediaPipe ASL Landmark Training Engine")
    print("-" * 60)
    
    try:
        path = kagglehub.dataset_download("ayuraj/asl-dataset")
        print(f"Dataset successfully located/downloaded at: {path}\n")
    except Exception as e:
        print(f"\n[!] ERROR downloading dataset automatically: {e}")
        sys.exit(1)

    # Resolve exact subfolder
    dataset_dir = os.path.join(path, "asl_dataset")
    if not os.path.exists(dataset_dir):
        paths = glob.glob(os.path.join(path, "*", "asl_dataset"))
        if paths:
            dataset_dir = paths[0]
        else:
            dataset_dir = path # fallback

    classes = list(string.ascii_lowercase)
    
    X = []
    y = []
    
    print("Initializing MediaPipe Hand Tracking for Feature Extraction...")
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )
    
    print("Extracting 3D geometry constraints from image dataset...")
    for label_idx, char_class in enumerate(classes):
        class_dir = os.path.join(dataset_dir, char_class)
        if not os.path.exists(class_dir):
            class_dir = os.path.join(dataset_dir, char_class.upper())
            
        if not os.path.exists(class_dir):
            print(f"[Warning] Could not find folder for class {char_class}. Skipping.")
            continue
            
        img_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.jpeg")) + glob.glob(os.path.join(class_dir, "*.png"))
        
        extracted_count = 0
        for img_path in img_paths:
            landmarks = extract_landmarks(img_path, mp_hands)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_idx)
                extracted_count += 1
                
        print(f"Class '{char_class.upper()}' -> Processed {extracted_count} robust signatures.")

    if len(X) == 0:
        print("[!] No landmarks could be extracted. Dataset might be empty or corrupted.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)
    
    print(f"\nExtracted {len(X)} total hand sign signatures.")
    
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y, num_classes=26)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    print("\nInitializing Dense Neural Architecture...")
    model = build_dnn_model()

    print("\nCommencing Advanced Landmark Model Training Phase...")
    target_weights = "sign_language_model.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        target_weights, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint]
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Landmark Neural Evaluation -> Test Accuracy: {accuracy*100:.2f}%")
    print(f"Your sophisticated topology writes have saved targeting '{target_weights}'.")

if __name__ == "__main__":
    main()
