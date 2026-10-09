import cv2
import threading
import queue
import time
import string
import numpy as np
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLOWorld
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import os
import uuid

# Online TTS + Translation
from gtts import gTTS
import pygame
from googletrans import Translator

MODEL_PATH = 'sign_language_model.h5'

speech_queue = queue.Queue()

try:
    pygame.mixer.init()
except Exception as e:
    print(f"[Audio] pygame init failed: {e}")

def tts_worker():
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        # SAPI5 for English (always offline)
        import win32com.client
        sapi = win32com.client.Dispatch("SAPI.SpVoice")
        sapi.Rate = 0
        sapi.Volume = 100
        print(f"[TTS] SAPI5 ready: {sapi.Voice.GetDescription()}")

        while True:
            item = speech_queue.get()
            if item is None:
                break

            text, target_lang = item
            tmp_file = None

            try:
                if target_lang == 'en':
                    # Fast offline English via SAPI
                    sapi.Speak(text)
                else:
                    # Step 1: Translate text
                    translator = Translator()
                    result = translator.translate(text, dest=target_lang)
                    translated_text = result.text
                    print(f"[TTS] Translated '{text}' -> '{translated_text}' ({target_lang})")

                    # Step 2: Speak translated text using gTTS (online)
                    tmp_file = f"tts_{uuid.uuid4().hex}.mp3"
                    tts = gTTS(text=translated_text, lang=target_lang, slow=False)
                    tts.save(tmp_file)

                    pygame.mixer.music.load(tmp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    pygame.mixer.music.unload()

            except Exception as err:
                print(f"[TTS] Error ({target_lang}): {err} — falling back to English SAPI")
                try:
                    sapi.Speak(text)
                except Exception:
                    pass
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except Exception:
                        pass

            speech_queue.task_done()

    except Exception as e:
        print(f"[TTS] Fatal error: {e}")

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def clear_audio_backlog():
    """Stops playback and drops all queued speech."""
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break

def speak(text, lang='en'):
    """Queues text for TTS. Drops if backlog is too large to prevent audio lag."""
    if speech_queue.qsize() < 2:
        speech_queue.put((text, lang))

class VideoCamera(object):
    def __init__(self):
        print("[!] Initializing VideoCamera Backend...")
        self.video = cv2.VideoCapture(0)
        
        self.mode = 'object_detection' 
        self.current_language = 'en'
        self.is_running = True
        self.frame_count = 0 
        
        print("[!] Warming up YOLO-World Open Vocabulary Topology...")
        self.yolo_model = YOLOWorld('yolov8s-worldv2.pt')
        
        # Expanding vocabulary to cover daily use items with precise descriptive names
        custom_classes = [
            # People & Body
            "person", "human hand",
            # Desk / Stationery
            "pen", "ballpoint pen", "pencil", "marker", "highlighter",
            "eraser", "sharpener", "ruler", "scissors",
            "notebook", "note", "notepad", "paper", "sticky note",
            "book", "textbook", "folder", "file",
            # Tech
            "computer mouse", "keyboard", "laptop", "monitor", "mouse pad",
            "cell phone", "mobile phone", "headphones", "earphones",
            "USB drive", "charger", "power bank",
            # Everyday items
            "bottle", "water bottle", "cup", "mug",
            "bag", "backpack", "wallet", "keys", "wrist watch",
            "spectacles", "glasses",
            "remote control", "calculator",
            # Furniture
            "chair", "table", "desk",
            # Misc
            "plant", "shoes",
        ]
        self.yolo_model.set_classes(custom_classes)
        
        self.is_model_untrained = False
        self.cnn_model = self.load_or_create_model()
        self.classes = list(string.ascii_uppercase)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.last_spoken = {}
        self.speak_cooldown = 4.0 

        # --- Finger-Spelling Sentence Architecture ---
        self.sentence_buffer = ""
        self.active_char = None
        self.char_lock_counter = 0
        self.hand_empty_frames = 0
        self.LOCK_THRESHOLD = 15 # 15 frames of stable math to lock character to UI
        self.SPACE_THRESHOLD = 30 # 30 frames (1s) of no hand -> Inserts "Space"
        self.SPEAK_THRESHOLD = 60 # 60 frames (2s) of no hand -> Audio Payout!

    def load_or_create_model(self):
        if os.path.exists(MODEL_PATH):
            return load_model(MODEL_PATH)
        else:
            self.is_model_untrained = True
            model = Sequential([
                Dense(128, activation='relu', input_shape=(42,)),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(26, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model

    def __del__(self):
        if self.video.isOpened():
            self.video.release()
        
    def stop(self):
        self.is_running = False
        if self.video.isOpened():
            self.video.release()

    def start(self):
        if not self.video.isOpened():
            self.video = cv2.VideoCapture(0)
        self.is_running = True

    def set_mode(self, mode):
        self.mode = mode
        self.last_spoken.clear()

    def set_language(self, language):
        self.current_language = language
        self.last_spoken.clear() 
        clear_audio_backlog() # Destroy the backlog so language switches INSTANTLY
        speak("Language changed.", language)

    def get_frame(self):
        if not self.is_running or not self.video.isOpened():
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "System Disabled / Offline", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes(), "Hardware stream paused."

        success, frame = self.video.read()
        if not success:
            return None, ""

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        self.frame_count += 1
        current_time = time.time()
        detected_text = ""

        if self.mode == 'object_detection':
            # Run Open Vocabulary Target Extraction globally across the entire space natively
            results = self.yolo_model(frame, stream=True, conf=0.25, verbose=False)
            detected_objs = set()

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.yolo_model.names[cls]
                    
                    detected_objs.add(class_name)

                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
            if detected_objs:
                detected_text = "[Object ID Active] Current targets: " + ", ".join(detected_objs)
            else:
                detected_text = "Scanning surroundings..."
            
            for obj in detected_objs:
                if obj not in self.last_spoken or (current_time - self.last_spoken[obj] > self.speak_cooldown):
                    speak(f"I see {obj}", self.current_language)
                    self.last_spoken[obj] = current_time

        elif self.mode == 'gesture_recognition':
            if self.is_model_untrained:
                cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.putText(frame, "WARNING: CNN UNTRAINED! MUST RUN train_sign_model.py FIRST", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            detected_text = "Awaiting structural hand signature..."

            if result.multi_hand_landmarks:
                self.hand_empty_frames = 0 # Hand traced, reset the organic sleep clock
                
                for hand_landmarks in result.multi_hand_landmarks:
                    hx_max, hy_max = 0, 0
                    hx_min, hy_min = w, h

                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > hx_max: hx_max = x
                        if x < hx_min: hx_min = x
                        if y > hy_max: hy_max = y
                        if y < hy_min: hy_min = y

                    w_box = hx_max - hx_min
                    h_box = hy_max - hy_min
                    max_dim = max(w_box, h_box)
                    
                    center_x = hx_min + w_box // 2
                    center_y = hy_min + h_box // 2
                    
                    padding = 10
                    new_hx_min = max(0, center_x - max_dim // 2 - padding)
                    new_hx_max = min(w, center_x + max_dim // 2 + padding)
                    new_hy_min = max(0, center_y - max_dim // 2 - padding)
                    new_hy_max = min(h, center_y + max_dim // 2 + padding)

                    cv2.rectangle(frame, (new_hx_min, new_hy_min), (new_hx_max, new_hy_max), (50, 255, 100), 2)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if new_hy_max > new_hy_min and new_hx_max > new_hx_min:
                        # Extract and Normalize Landmarks relative to wrist
                        wrist = hand_landmarks.landmark[0]
                        coords = []
                        for lm in hand_landmarks.landmark:
                            # [CHIRALITY FIX] The webcam feed is mirrored horizontally!
                            # We invert the X coordinate calculation to mathematically restore the right-hand geometry
                            # that the dataset was trained on.
                            coords.append(wrist.x - lm.x) 
                            coords.append(lm.y - wrist.y)
                            
                        coords = np.array(coords)
                        max_val = np.max(np.abs(coords))
                        if max_val > 0:
                            coords = coords / max_val
                            
                        # Reshape to (1, 42) for Dense Neural Network
                        coords_reshaped = np.expand_dims(coords, axis=0)

                        prediction_tensor = self.cnn_model(coords_reshaped, training=False)
                        prediction_array = prediction_tensor.numpy()[0]
                        predicted_class_idx = np.argmax(prediction_array)
                        confidence = float(prediction_array[predicted_class_idx])
                        predicted_char = self.classes[predicted_class_idx]

                        label_text = f'Class: {predicted_char} | Conf: {confidence:.2f}'
                        cv2.putText(frame, label_text, (new_hx_min, new_hy_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                        
                        detected_text = f"[Semantic Gesture Detected] : '{predicted_char}' -> ({confidence*100:.1f}%)"

                        # [NEW] Organic Sentence Flow (Replaces screaming single letters)
                        if confidence > 0.65 and not self.is_model_untrained:  
                            if predicted_char == self.active_char:
                                if self.char_lock_counter < self.LOCK_THRESHOLD:
                                    self.char_lock_counter += 1
                                    if self.char_lock_counter == self.LOCK_THRESHOLD:
                                        self.sentence_buffer += predicted_char
                            else:
                                self.active_char = predicted_char
                                self.char_lock_counter = 1
                        else:
                            self.active_char = None # Breaks confidence lock allowing for double-letters natively!

            else:
                self.hand_empty_frames += 1
                if self.hand_empty_frames == self.SPACE_THRESHOLD:
                    if self.sentence_buffer and not self.sentence_buffer.endswith(" "):
                        self.sentence_buffer += " "
                
                if self.hand_empty_frames == self.SPEAK_THRESHOLD:
                    if self.sentence_buffer.strip():
                        speak(self.sentence_buffer.strip(), self.current_language)
                        self.sentence_buffer = "" # Purges the string structure instantly 

            # Overlay tracking text smoothly physically pinned onto the video stream bottom bound
            cv2.putText(frame, f"Spelling: {self.sentence_buffer}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), detected_text
