import os
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
 
prompt_responses = {}
 
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FRAMES_FOLDER'] = 'processed_frames'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
 
# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FRAMES_FOLDER'], exist_ok=True)
os.makedirs('anomalous_clips', exist_ok=True)  # Folder for video clips
 
# Load environment variables
load_dotenv()
genai.configure(api_key='AIzaSyCRU31GS3v7eiqXLPR4gAKRigbIB2i_L4E')
 
# Load autoencoder model
model = tf.keras.models.load_model("autoencoder_video_complex.h5")
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
 
# Function to clear the processed frames folder
def clear_processed_frames_folder():
    for filename in os.listdir(app.config['PROCESSED_FRAMES_FOLDER']):
        file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
 
# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    return frame
 
# Function to detect anomalies
def detect_anomaly(autoencoder, frame):
    reconstructed = autoencoder.predict(frame)
    mse = np.mean(np.power(frame - reconstructed, 2))
    threshold = 0.0235
    return mse > threshold
 
# Function to save video clip using moviepy
def save_video_clip(video_path, start_frame, end_frame, output_clip_path):
    video = VideoFileClip(video_path)
    start_time = start_frame / video.fps
    end_time = end_frame / video.fps
    clip = video.subclip(start_time, end_time)
    clip.write_videofile(output_clip_path, codec='libx264')
 
def get_gemini_video_narration(video_path):
    video_file = genai.upload_file(path=video_path)
    prompt = "Describe the possible anomaly in this image in a single sentence"
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
    genai.delete_file(video_file.name)
 
    return response.text
 
def process_video(video_path, output_dir_clip):
    clear_processed_frames_folder()
    cap = cv2.VideoCapture(video_path)
    i = 0
    warm_up_frames = 60
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        preprocessed_frame = preprocess_frame(frame)
       
        if i > warm_up_frames:
            if detect_anomaly(model, preprocessed_frame):  
                # Save video clip of the anomaly
                start_frame = max(0, i - 25)
                end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), i + 600)  
                clip_path = os.path.join(output_dir_clip, f"anomalous_clip_0.mp4")
                save_video_clip(video_path, start_frame, end_frame, clip_path)
               
                # Narrate video clip using Gemini model
                narration = get_gemini_video_narration(clip_path)
                prompt_responses[f'anomalous_clip_0.mp4'] = narration
                break      
        i += 1
   
    cap.release()
    cv2.destroyAllWindows()
 
# @app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            # Start processing the video in the background
            from threading import Thread
            def process():
                output_dir_clip = 'anomalous_clips'
                process_video(video_path, output_dir_clip)
            
            thread = Thread(target=process)
            thread.start()
            
            return render_template('loading.html')
    return render_template('upload1.html')

# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(video_path)
#             output_dir_clip = 'anomalous_clips'
#             process_video(video_path, output_dir_clip)
#             return redirect(url_for('show_anomalous_clips'))
#     return render_template('upload2.html')
 
@app.route('/anomalous_clips/<filename>')
def anomalous_clips(filename):
    return send_from_directory('anomalous_clips', filename)
 
@app.route('/anomalous_clips', methods=['GET'])
def show_anomalous_clips():
    clips_info = []
    for filename in os.listdir('anomalous_clips'):
        if filename.endswith('.mp4'):
            description = prompt_responses.get(filename, 'Description not available.')
            clips_info.append({'filename': filename, 'description': description})
   
    if not clips_info:
        return render_template('no_clips.html')
   
    return render_template('anomalous_clips.html', clips=clips_info)
 
 
if __name__ == '__main__':
    app.run(debug=True)