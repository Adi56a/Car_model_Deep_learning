import os
from flask import Flask, render_template, request, send_file, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Your inference methods (adjust import as per your project layout)
from inference.detector import run_inference         # YOLO-P lane detection
from yolo_processer import process_video, process_image  # Object detection

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
INFERENCE_INPUT = os.path.join(BASE_DIR, 'inference_input')
INFERENCE_OUTPUT = os.path.join(BASE_DIR, 'inference_output')

# Initialize app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(INFERENCE_INPUT, exist_ok=True)
os.makedirs(INFERENCE_OUTPUT, exist_ok=True)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/options')
def options():
    return render_template('option.html')

@app.route('/lane-detection', methods=['GET', 'POST'])
def lane_detection():
    result_url = None
    download_filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(INFERENCE_INPUT, filename)
            file.save(input_path)

            # Run YOLO-P lane detection
            result_path = run_inference(input_path, INFERENCE_OUTPUT)
            result_filename = os.path.basename(result_path)

            # Move to static folder for web access
            static_path = os.path.join(STATIC_FOLDER, result_filename)
            os.replace(result_path, static_path)

            result_url = url_for('static', filename=result_filename)
            download_filename = result_filename

    return render_template('lanefile.html', result=result_url, download=download_filename)

@app.route('/download/<filename>')
def download_lane_result(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename), as_attachment=True)


@app.route('/od_video')
def od_video():
    return render_template('object_detection_video.html')

@app.route('/od-video-upload', methods=['POST'])
def upload_video():
    video = request.files['media']
    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(video_path)

        output_path = process_video(video_path)
        return jsonify({"success": True, "path": "/" + output_path})
    return jsonify({"success": False}), 400

@app.route('/od-image-upload', methods=['POST'])
def upload_image():
    image = request.files['media']
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)

        output_path = process_image(image_path)
        return jsonify({"success": True, "path": "/" + output_path})
    return jsonify({"success": False}), 400

@app.route('/od-video-download')
def download_video():
    return send_from_directory(directory=STATIC_FOLDER, path="output_video.mp4", as_attachment=True)

@app.route('/od-image-download')
def download_image():
    return send_from_directory(directory=STATIC_FOLDER, path="output_image.jpg", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
