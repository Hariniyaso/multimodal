from flask import Flask, render_template, Response, jsonify, request
import time
from camera import VideoCamera

app = Flask(__name__)

camera = None
latest_output = "System Initializing. Preparing ML Topology..."

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

@app.route('/')
def index():
    return render_template('index.html')

def gen(local_camera):
    global latest_output
    while True:
        if local_camera is not None and local_camera.is_running:
            frame, detected_text = local_camera.get_frame()
            if detected_text:
                latest_output = detected_text
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    cam = get_camera()
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    data = request.json
    action = data.get('action')
    cam = get_camera()
    
    if action == 'start':
        cam.start()
        return jsonify({"status": "Camera started"})
    elif action == 'stop':
        cam.stop()
        return jsonify({"status": "Camera stopped"})
    
    return jsonify({"error": "Invalid action context"}), 400

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    mode = data.get('mode')
    cam = get_camera()
    
    if mode in ['object_detection', 'gesture_recognition']:
        cam.set_mode(mode)
        sanitized = mode.replace('_', ' ')
        
        # Bypass direct speak calls here since camera will announce dynamically
        return jsonify({"status": f"Mode mapping redefined to '{mode}'"})
        
    return jsonify({"error": "System architecture prohibits specified target module routing"}), 400

@app.route('/set_language', methods=['POST'])
def set_language():
    """Maps the language selection dynamically directly into the processing pipeline"""
    data = request.json
    language = data.get('language')
    cam = get_camera()
    
    # Valid targets specified
    if language in ['en', 'hi', 'ta']:
        cam.set_language(language)
        
        langs = {'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil'}
        human_lang = langs.get(language, language)
        
        return jsonify({"status": f"Language mapped to {human_lang}"})
        
    return jsonify({"error": "Unsupported layout target localization"}), 400

@app.route('/get_output')
def get_output():
    global latest_output
    return jsonify({"output": latest_output})

if __name__ == '__main__':
    print("[INIT] Triggering early Neural topology preload sequence...")
    get_camera()
    # Explicitly disabling werkzeug reload thread lock for cleaner pygame executions
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
