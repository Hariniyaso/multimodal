import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import time

# Initialize speech queue to handle TTS asynchronously
speech_queue = queue.Queue()

def tts_worker():
    """Worker thread to handle text-to-speech without blocking the video stream."""
    try:
        # Initialize the engine in the worker thread to避免 issues on Windows (SAPI5)
        engine = pyttsx3.init()
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
    """Add text to the speech queue."""
    speech_queue.put(text)

def main():
    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    print("Loading YOLOv8 model...")
    # Load the pretrained YOLOv8 model (yolov8n is the fastest/smallest)
    model = YOLO('yolov8n.pt')

    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Track when objects were last spoken to avoid spamming the audio output
    last_spoken = {}
    speak_cooldown = 3.0  # seconds between repeated announcements for the same object

    print("Starting real-time object detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Run inference (set conf=0.5 to apply the confidence threshold)
        results = model(frame, stream=True, conf=0.5)
        
        detected_objects = set()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence score
                conf = float(box.conf[0])

                # Get class label
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Add to set of detected objects in the current frame
                detected_objects.add(class_name)

                # Draw bounding box and label on the frame
                label = f'{class_name} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        
        for obj in detected_objects:
            # Print detected object name
            print(f"Detected: {obj}")
            
            # Announce via TTS if the object hasn't been announced recently
            if obj not in last_spoken or (current_time - last_spoken[obj] > speak_cooldown):
                speak(f"I see a {obj}")
                last_spoken[obj] = current_time

        # Display the frame with bounding boxes
        cv2.imshow('Yolov8 Real-Time Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Send None to stop the TTS worker thread gracefully
    speech_queue.put(None)
    tts_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
