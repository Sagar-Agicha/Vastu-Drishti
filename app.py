# Training Time - 
# Testing Time - 7 mins and 9 secs
import zipfile
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file, session
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import cv2
from collections import OrderedDict
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import yaml
import numpy as np
from datetime import datetime
import pytesseract
import aiofiles
from openpyxl import Workbook
from time import time
import io
import logging
import uuid
from pdf2image import convert_from_path
import tempfile

app = Flask(__name__)
app.secret_key = 'SAGAR'  # Add this line

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Add with other configurations
DOWNLOAD_FOLDER = os.path.join(app.static_folder, 'downloads')
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the upload folders with absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads', 'images')
VIDEO_FOLDER = os.path.join(BASE_DIR, 'uploads', 'videos')

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Add these global variables
LABELS_FILE = 'labels.txt'
ANNOTATIONS_FOLDER = 'annotations'
DATASET_ROOT = 'dataset'

# Create annotations directory
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)

# Initialize or load class labels
def init_labels():
    try:
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
                return OrderedDict((label, idx) for idx, label in enumerate(labels) if label)
        return OrderedDict()
    except Exception as e:
        print(f"Error initializing labels: {e}")
        return OrderedDict()

class_labels = init_labels()

@app.route('/debug_labels')
def debug_labels():
    try:
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            return jsonify({
                'labels_file_exists': True,
                'labels': labels,
                'class_labels': list(class_labels.items())
            })
        return jsonify({
            'labels_file_exists': False,
            'error': 'Labels file does not exist'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov', 'mkv', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def login():
    return render_template('login.html')
 
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.debug("Starting file upload")
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'success': False, 'error': 'No file part'}), 400

        files = request.files.getlist('file')
        uploaded_files = []

        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                # Generate UUID-based unique ID
                unique_id = str(uuid.uuid4().hex[:8])  # Using first 8 characters of UUID
                original_filename = secure_filename(file.filename)
                filename = f"{unique_id}_{original_filename}"
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logger.debug(f"Saving file to: {file_path}")
                
                file.save(file_path)
                uploaded_files.append(filename)
                logger.debug(f"File saved successfully: {filename}")

        if uploaded_files:
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded {len(uploaded_files)} files',
                'files': uploaded_files
            })
        else:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400

    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_images')
def get_images():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            files.append({
                'name': filename,
                'url': f'/uploads/{filename}'
            })
    return jsonify({'images': files})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        video = request.files['video']
        if video.filename == '':
            return jsonify({'success': False, 'error': 'No video file selected'}), 400
        if video and allowed_file(video.filename):
            # Generate unique ID using uuid
            unique_id = str(uuid.uuid4().hex[:8])
            original_filename = secure_filename(video.filename)
            filename = f"{unique_id}_{original_filename}"
            
            video_path = os.path.join(VIDEO_FOLDER, filename)
            logger.debug(f"Saving video to: {video_path}")
            
            video.save(video_path)
            
            if os.path.exists(video_path):
                logger.debug(f"Video saved successfully at {video_path}")
                return jsonify({ 
                    'success': True,
                    'filename': filename,
                    'message': 'Video uploaded successfully'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to save video file'}), 500
                
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        data = request.get_json()
        filename = data.get('filename')
        interval = data.get('interval')
        
        logger.debug(f"Processing video request - filename: {filename}, interval: {interval}")
        
        if not filename or not interval:
            return jsonify({'success': False, 'error': 'Missing filename or interval'}), 400
            
        video_path = os.path.join(VIDEO_FOLDER, filename)
        logger.debug(f"Looking for video at: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found at {video_path}")
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
            
        # Process video by time interval
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"{os.path.splitext(filename)[0]}_frame_{saved_count}.jpg"
                frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        
        return jsonify({
            'success': True,
            'message': f'Extracted {saved_count} frames at {interval} second intervals'
        })
        
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process_video_by_count', methods=['POST'])
def process_video_by_count():
    try:
        data = request.get_json()
        filename = data.get('filename')
        total_frames = data.get('total_frames')
        
        if not filename or not total_frames:
            return jsonify({'success': False, 'error': 'Missing filename or total_frames'}), 400
            
        # Get video from VIDEO_FOLDER
        video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
            
        # Process video by frame count
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = total_video_frames // total_frames
        base_filename = os.path.splitext(filename)[0]
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0 and saved_count < total_frames:
                # Save frames to UPLOAD_FOLDER (image folder)
                frame_filename = f"{base_filename}_frame_{saved_count}.jpg"
                frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        
        # Delete the video file after processing
        try:
            os.remove(video_path)
            print(f"Successfully deleted video: {video_path}")
        except Exception as e:
            print(f"Error deleting video {video_path}: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'Extracted {saved_count} frames evenly distributed throughout the video'
        })
        
    except Exception as e:
        print(f"Error in process_video_by_count: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def extract_frames(video_path, output_dir, skip_frames=2):
    """Extract frames from video by skipping specified number of frames"""
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's at the skip interval
        if frame_count % skip_frames == 0:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return saved_count

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.get_json()
        logger.debug(f"Received annotation data: {data}")  # Debug log
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        image_file = data.get('image_file')
        annotations = data.get('annotations', [])
        
        if not image_file:
            return jsonify({'success': False, 'error': 'Missing image_file'}), 400

        # Ensure ANNOTATIONS_FOLDER exists
        os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
        
        # Get base filename without extension
        base_filename = os.path.splitext(image_file)[0]
        
        # Create a set of existing labels
        existing_labels = set()
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                existing_labels = set(line.strip() for line in f.readlines())
        
        # Collect new labels
        new_labels = set()
        for ann in annotations:
            class_name = ann.get('class')
            if class_name and class_name not in existing_labels:
                new_labels.add(class_name)
        
        # Update labels file if there are new labels
        if new_labels:
            all_labels = sorted(existing_labels.union(new_labels))
            with open(LABELS_FILE, 'w') as f:
                for label in all_labels:
                    f.write(f"{label}\n")
            
            # Update class_labels dictionary
            global class_labels
            class_labels = OrderedDict((label, idx) for idx, label in enumerate(all_labels))
        
        # Create YOLO format annotations
        yolo_annotations = []
        for ann in annotations:
            class_name = ann.get('class')
            if not class_name or class_name not in class_labels:
                continue
                
            class_idx = class_labels[class_name]
            x = float(ann.get('x', 0))
            y = float(ann.get('y', 0))
            w = float(ann.get('width', 0))
            h = float(ann.get('height', 0))
            
            yolo_annotations.append(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        
        # Save annotations to file
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
        with open(annotation_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        return jsonify({
            'success': True,
            'message': 'Annotations saved successfully',
            'annotations_path': annotation_path
        })
        
    except Exception as e:
        logger.error(f"Error in save_annotations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_annotations/<image_file>')
def get_annotations(image_file):
    try:
        base_filename = os.path.splitext(image_file)[0]
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
        
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # class_idx x y width height
                        class_idx = int(parts[0])
                        # Find class name from index
                        class_name = next((k for k, v in class_labels.items() if v == class_idx), 'unknown')
                        annotations.append({
                            'class': class_name,
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        
        return jsonify({'annotations': annotations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def add_noise(image):
    # Gaussian Noise
    def gaussian_noise(img):
        mean = 0
        sigma = 25
        noise = np.random.normal(mean, sigma, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Salt and Pepper Noise
    def salt_pepper_noise(img):
        prob = 0.05
        noisy = np.copy(img)
        # Salt
        salt_mask = np.random.random(img.shape) < prob/2
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(img.shape) < prob/2
        noisy[pepper_mask] = 0
        return noisy

    return [
        gaussian_noise(image),
        salt_pepper_noise(image)
    ]

@app.route('/prepare_training', methods=['POST'])
def prepare_training():
    try:
        # Create dataset structure if it doesn't exist
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(DATASET_ROOT, split)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

        # Get existing class names
        class_names = {}
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                class_names = {i: name.strip() for i, name in enumerate(f.readlines())}

        # Get all new images and their annotations
        image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        # Split new data
        train_files, temp_files = train_test_split(image_files, train_size=0.7, random_state=42)
        valid_files, test_files = train_test_split(temp_files, train_size=0.67, random_state=42)

        def append_files_with_augmentation(files, split):
            saved_count = 0
            # Get the current highest index in the split directory
            split_dir = os.path.join(DATASET_ROOT, split, 'images')
            existing_files = [f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            start_index = len(existing_files)

            for idx, img_file in enumerate(files, start=start_index):
                # Copy original image
                src_img = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]

                # Read original image
                img = cv2.imread(src_img)
                if img is None:
                    continue

                # Save original image with new index
                dst_img = os.path.join(DATASET_ROOT, split, 'images', f"image_{idx:06d}{ext}")
                cv2.imwrite(dst_img, img)
                saved_count += 1

                # Copy original annotation if exists
                ann_file = f"{base_name}.txt"
                src_ann = os.path.join(ANNOTATIONS_FOLDER, ann_file)
                if os.path.exists(src_ann):
                    dst_ann = os.path.join(DATASET_ROOT, split, 'labels', f"image_{idx:06d}.txt")
                    shutil.copy2(src_ann, dst_ann)

                # Generate and save augmented images
                noisy_images = add_noise(img)
                for aug_idx, noisy_img in enumerate(noisy_images):
                    aug_img_path = os.path.join(DATASET_ROOT, split, 'images', 
                                              f"image_{idx:06d}_aug{aug_idx}{ext}")
                    cv2.imwrite(aug_img_path, noisy_img)
                    saved_count += 1

                    # Copy annotation for augmented image
                    if os.path.exists(src_ann):
                        aug_ann_path = os.path.join(DATASET_ROOT, split, 'labels',
                                                  f"image_{idx:06d}_aug{aug_idx}.txt")
                        shutil.copy2(src_ann, aug_ann_path)

            return saved_count

        # Append files to respective splits with augmentation
        train_count = append_files_with_augmentation(train_files, 'train')
        valid_count = append_files_with_augmentation(valid_files, 'valid')
        test_count = append_files_with_augmentation(test_files, 'test')

        # Update data.yaml file
        yaml_content = {
            'path': os.path.abspath(DATASET_ROOT),
            'train': os.path.join('train', 'images'),
            'val': os.path.join('valid', 'images'),
            'test': os.path.join('test', 'images'),
            'nc': len(class_names),
            'names': [class_names[i] for i in range(len(class_names))]
        }

        yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        return jsonify({
            'success': True,
            'message': f'Dataset updated with {train_count} training, {valid_count} validation, and {test_count} test images (including augmentations)'
        })

    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_annotations', methods=['POST'])
def clear_annotations():
    try:
        data = request.json
        image_file = data['image_file']
        
        # Get base filename without extension
        base_filename = os.path.splitext(image_file)[0]
        
        # Clear annotation file
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_filename}.txt")
        if os.path.exists(annotation_path):
            # Write empty file
            open(annotation_path, 'w').close()
        
        return jsonify({'success': True, 'message': 'Annotations cleared successfully'})
    
    except Exception as e:
        print(f"Error clearing annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all_annotations', methods=['POST'])
def clear_all_annotations():
    try:
        # Get absolute path to annotations folder
        annotations_path = os.path.abspath(ANNOTATIONS_FOLDER)
        print(f"Clearing annotations from: {annotations_path}")  # Debug print
        
        # Get all files in the annotations directory
        annotation_files = [f for f in os.listdir(annotations_path) 
                          if f.endswith('.txt')]
        
        print(f"Found {len(annotation_files)} annotation files to delete")  # Debug print
        
        deleted_count = 0
        for ann_file in annotation_files:
            file_path = os.path.join(annotations_path, ann_file)
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)  # Using unlink instead of remove
                    deleted_count += 1
                    print(f"Deleted: {file_path}")  # Debug print
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
                continue  # Continue with next file even if one fails
        
        # Also delete labels.txt if it exists
        labels_path = os.path.abspath(LABELS_FILE)
        if os.path.exists(labels_path):
            try:
                os.unlink(labels_path)
                print(f"Deleted labels file: {labels_path}")  # Debug print
            except Exception as e:
                print(f"Error deleting labels file: {str(e)}")
        
        # Reset class_labels global variable
        global class_labels
        class_labels = OrderedDict()
        
        print(f"Successfully deleted {deleted_count} annotation files")  # Debug print
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} annotation files'
        })
    
    except Exception as e:
        error_msg = f"Error clearing annotations: {str(e)}"
        print(error_msg)  # Debug print
        return jsonify({
            'error': error_msg,
            'success': False
        }), 500

@app.route('/get_video_details', methods=['POST'])
def get_video_details():
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
            
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return jsonify({
            'success': True,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_images', methods=['POST'])
def clear_images():
    try:
        # Clear images from upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        return jsonify({
            'success': True,
            'message': 'All images cleared successfully'
        })
    except Exception as e:
        print(f"Error clearing images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_process', methods=['POST'])
def start_process():
    try:
        # Initialize training process
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add these constants at the top of your file with other configurations
MODEL_DIR = 'runs/detect/train'
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/training_progress')
def training_progress():
    try:
        start_time = time()
        
        dataset_root = os.path.abspath(DATASET_ROOT)
        yaml_path = os.path.join(dataset_root, 'data.yaml')
        
        # Check for existing model
        existing_model_path = os.path.join(MODEL_DIR, 'weights/last.pt')
        if os.path.exists(existing_model_path):
            print(f"Using existing model: {existing_model_path}")
            model = YOLO(existing_model_path)
        else:
            print("No existing model found, using pretrained model")
            model = YOLO("yolo11m.pt")
        
        results = model.train(
            data=yaml_path,
            epochs=100,
            project='runs/detect',
            name='train',
            exist_ok=True
        )
        
        training_time = time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        model_path = os.path.join(MODEL_DIR, 'weights/last.pt')

        print(f"Training completed in {hours}h {minutes}m {seconds}s")
        
        if os.path.exists(model_path):
            print(f"Model saved at: {model_path}")
            return jsonify({
                'success': True, 
                'message': f'Training completed in {hours}h {minutes}m {seconds}s', 
                'model_path': model_path,
                'training_time': {
                    'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds,
                    'total_seconds': training_time
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Model file not found'}), 500
        
    except Exception as e:
        print(f"Error in training_progress: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_class', methods=['POST'])
def save_class():
    try:
        print("Received save_class request")  # Debug print
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        class_name = data.get('class_name')
        print(f"Class name received: {class_name}")  # Debug print
        
        if not class_name:
            return jsonify({'error': 'No class name provided'}), 400

        # Read existing classes
        existing_classes = set()
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                existing_classes = set(line.strip() for line in f)

        # Add new class if it doesn't exist
        if class_name not in existing_classes:
            with open(LABELS_FILE, 'a') as f:
                f.write(f"{class_name}\n")
            print(f"Added new class: {class_name}")  # Debug print

        return jsonify({'success': True, 'message': f'Class {class_name} saved successfully'})
    except Exception as e:
        print(f"Error in save_class: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

# Add these global variables at the top with other configurations
TEST_FOLDER = os.path.join(BASE_DIR, 'uploads', 'test')
os.makedirs(TEST_FOLDER, exist_ok=True)

@app.route('/get_test_file_count')
def get_test_file_count():
    try:
        # Initialize counters
        counts = {
            'images': 0,
            'videos': 0,
            'pdfs': 0
        }
        
        # Count files in test folder
        if os.path.exists(TEST_FOLDER):
            for file in os.listdir(TEST_FOLDER):
                lower_file = file.lower()
                if lower_file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    counts['images'] += 1
                elif lower_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    counts['videos'] += 1
                elif lower_file.endswith('.pdf'):
                    counts['pdfs'] += 1
        
        return jsonify({
            'success': True,
            'counts': counts
        })
        
    except Exception as e:
        logger.error(f"Error getting test file count: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add this endpoint for handling test file uploads
@app.route('/upload_test_files', methods=['POST'])
def upload_test_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        uploaded_files = []
        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                # Create unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                random_id = str(uuid.uuid4())[:8]
                filename = f"test_{timestamp}_{random_id}_{secure_filename(file.filename)}"
                
                # Save file
                filepath = os.path.join(TEST_FOLDER, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                logger.info(f"Saved test file: {filename}")

        if uploaded_files:
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded {len(uploaded_files)} files',
                'files': uploaded_files
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No valid files were uploaded'
            }), 400

    except Exception as e:
        logger.error(f"Error uploading test files: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def draw_validation_image(original_img, annotated_img, predictions, confidence_threshold=0.6):
    """Create a side-by-side comparison image with detailed information"""
    
    # Convert BGR to RGB if needed
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Create a larger canvas for side-by-side comparison
    h, w = original_img.shape[:2]
    validation_img = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
    
    # Place original and annotated images
    validation_img[:, :w] = original_img
    validation_img[:, w+20:] = annotated_img
    
    # Add dividing line
    cv2.line(validation_img, (w+10, 0), (w+10, h), (255, 255, 255), 2)
    
    # Add labels
    cv2.putText(validation_img, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(validation_img, "Annotated", (w+30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add detection information
    info_y = 60
    for pred in predictions:
        if pred['confidence'] >= confidence_threshold:
            info = f"Class: {pred['class']}, Conf: {pred['confidence']:.2f}"
            cv2.putText(validation_img, info, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += 20
    
    return validation_img

@app.route('/start_testing', methods=['POST'])
async def start_testing():
    try:
        data = request.json
        mode = data.get('mode', 'ocr')
        
        # Check TEST_FOLDER directly
        test_files = []
        if os.path.exists(TEST_FOLDER):
            test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
            print(f"Found {len(test_files)} files in {TEST_FOLDER}")  # Debug print
        
        if not test_files:
            print(f"No files found in {TEST_FOLDER}")  # Debug print
            return jsonify({'error': 'No test files found', 'folder': TEST_FOLDER}), 400

        # Get all classes from the model
        model_path = os.path.join(MODEL_DIR, 'weights/last.pt')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found. Please train the model first.'}), 404
            
        model = YOLO(model_path)
        all_classes = model.names

        # Process files and collect results
        results = {}
        for file_path in test_files:
            print(f"Processing file: {file_path}")  # Debug print
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                filename, file_results = await process_video_with_ocr(file_path, model, all_classes)
            else:
                filename, file_results = await process_image_with_ocr(file_path, model, all_classes)

            if filename and file_results:
                results[filename] = file_results

        # Store results in session for download
        session['ocr_results'] = results

        return jsonify({
            'success': True,
            'message': f'Testing completed successfully. Processed {len(test_files)} files.',
            'results': results
        })

    except Exception as e:
        print(f"Error in start_testing: {str(e)}")
        return jsonify({'error': str(e)}), 500

async def process_image_with_ocr(image_path, model, all_classes):
    """Process a single image with OCR for all detected classes"""
    try:
        CONFIDENCE_THRESHOLD = 0.6
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Get predictions
        results = model(img)
        detected_data = {}
        filename = os.path.basename(image_path)

        # Save annotated image in the correct directory
        filename = os.path.basename(image_path)
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'annotated_{filename}')
        cv2.imwrite(output_path, img)

        # Process each detection
        for result in results[0].boxes:
            if result.conf[0].item() < CONFIDENCE_THRESHOLD:
                continue

            # Get coordinates and class
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            class_id = int(result.cls[0])
            class_name = model.names[class_id]
            
            # Crop detected region
            crop_img = img[y1:y2, x1:x2]
            
            # Perform OCR on cropped region
            ocr_text = run_ocr_on_crop(crop_img)
            
            # Store results
            if class_name not in detected_data:
                detected_data[class_name] = []
            detected_data[class_name].append(ocr_text)

            # Draw bounding box for visualization
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name}: {ocr_text}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save annotated image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated', filename)
        cv2.imwrite(output_path, img)

        return filename, detected_data

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None
    
async def process_video_with_ocr(video_path, model, all_classes, interval_seconds=0.2):
    """Process video frames with OCR for all detected classes at specified time intervals"""
    try:
        CONFIDENCE_THRESHOLD = 0.6
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval based on seconds
        frame_interval = int(fps * interval_seconds)
        
        # Setup video writer with correct output path
        filename = os.path.basename(video_path)
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'annotated_{filename}')
        
        # If input is MP4, ensure output is also MP4
        if filename.lower().endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            # For other formats, use a default codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        detected_data = {}
        current_time = 0.0

        while frame_count < total_frames:
            print(f"Processing frame {frame_count} of {total_frames}")  # Debug print
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame only at specified interval
            if frame_count % frame_interval == 10:
                # Calculate current time in seconds
                current_time = frame_count / fps
                
                # Process frame
                results = model(frame)
                frame_data = {}

                # Process each detection in the frame
                for result in results[0].boxes:
                    if result.conf[0].item() < CONFIDENCE_THRESHOLD:
                        continue

                    # Get coordinates and class
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    class_id = int(result.cls[0])
                    class_name = model.names[class_id]
                    
                    # Crop detected region
                    crop_img = frame[y1:y2, x1:x2]
                    
                    # Perform OCR on cropped region
                    ocr_text = run_ocr_on_crop(crop_img)
                    
                    # Store results
                    if class_name not in frame_data:
                        frame_data[class_name] = []
                    frame_data[class_name].append(ocr_text)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name}: {ocr_text}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store frame results with timestamp
                if frame_data:
                    detected_data[f"time_{current_time:.1f}s"] = frame_data

            # Write frame (whether processed or not)
            out.write(frame)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        return filename, detected_data

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None, None
@app.route('/download_results', methods=['GET'])
def download_results():
    try:
        # Create a temporary directory for organizing files
        temp_dir = os.path.join(app.static_folder, 'temp_download')
        os.makedirs(temp_dir, exist_ok=True)

        # Create subdirectories
        excel_dir = os.path.join(temp_dir, 'excel_results')
        media_dir = os.path.join(temp_dir, 'annotated_media')
        os.makedirs(excel_dir, exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)

        # Create Excel file
        wb = Workbook()
        ws = wb.active
        ws.title = "Detection Results"
        
        # Get results from the session
        results = session.get('ocr_results', {})
        
        # Get all unique classes from the results
        all_classes = set()
        for file_data in results.values():
            if isinstance(file_data, dict):
                for frame_data in file_data.values():
                    all_classes.update(frame_data.keys())
        
        # Sort classes to maintain consistent order
        all_classes = sorted(list(all_classes))
        
        # Create headers: Frame/Time, followed by each class
        headers = ['Frame/Time']
        headers.extend([f"Class_{class_name}_OCR" for class_name in all_classes])
        ws.append(headers)
        
        # Process and write results
        for filename, file_data in results.items():
            if isinstance(file_data, dict):
                # Sort frames/timestamps to ensure chronological order
                frames = sorted(file_data.keys())
                
                for frame in frames:
                    row_data = [frame]  # Start with frame/timestamp
                    frame_results = file_data[frame]
                    
                    # Add OCR results for each class
                    for class_name in all_classes:
                        ocr_texts = frame_results.get(class_name, [])
                        row_data.append('; '.join(ocr_texts) if ocr_texts else 'N/A')
                    
                    ws.append(row_data)
        
        # Save Excel file
        excel_path = os.path.join(excel_dir, 'detection_results.xlsx')
        wb.save(excel_path)

        # Copy annotated media files based on original file type
        annotated_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated')
        if os.path.exists(annotated_folder):
            for filename in os.listdir(annotated_folder):
                src_path = os.path.join(annotated_folder, filename)
                dst_path = os.path.join(media_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)

        # Create ZIP file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f'test_results_{timestamp}.zip'
        zip_path = os.path.join(app.static_folder, 'downloads', zip_filename)
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files from temp directory to ZIP
            for folder_path, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_path, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )

    except Exception as e:
        print(f"Error creating download package: {str(e)}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({'error': str(e)}), 500

@app.route('/download_results/<filename>')
def download_file(filename):
    return send_from_directory(os.path.join(app.static_folder, 'output'), filename)

@app.route('/download_all_results')
def download_all_results():
    try:
        # Create necessary directories if they don't exist
        output_dir = os.path.join(app.static_folder, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Get the latest session info
        if 'LATEST_SESSION' not in app.config:
            return jsonify({'error': 'No test results available'}), 404
            
        session_info = app.config['LATEST_SESSION']
        session_folder = session_info['folder']

        # Create a temporary directory for organizing files
        temp_dir = os.path.join(output_dir, 'temp_zip')
        os.makedirs(temp_dir, exist_ok=True)

        # Create subdirectories in temp folder
        excel_dir = os.path.join(temp_dir, 'excel_results')
        media_dir = os.path.join(temp_dir, 'annotated_media')
        os.makedirs(excel_dir, exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)

        # Copy all files from the session folder to temp directory
        for root, _, files in os.walk(session_folder):
            for file in files:
                src_path = os.path.join(root, file)
                # Determine destination based on file type
                if file.endswith('.xlsx'):
                    dst_path = os.path.join(excel_dir, file)
                else:  # Media files (images and videos)
                    dst_path = os.path.join(media_dir, file)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)

        # Create the ZIP file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = os.path.join(output_dir, f'all_results_{timestamp}.zip')

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add files from temp directory to ZIP
            for folder_path, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_path, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        # Send the ZIP file
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'all_results_{timestamp}.zip'
        )

    except Exception as e:
        print(f"Error creating ZIP file: {str(e)}")
        return jsonify({'error': str(e)}), 500

async def async_read_image(path):
    """Async function to read an image"""
    async with aiofiles.open(path, mode='rb') as f:
        img_bytes = await f.read()
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def run_ocr_on_crop(crop_img):
    """OCR on cropped image"""
    if crop_img is None or crop_img.size == 0:
        return ""
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray_img).strip()

@app.route('/delete_project', methods=['POST'])
def delete_project():
    try:
        # Define test directory
        TEST_FOLDER = 'static/test'
        os.makedirs(TEST_FOLDER, exist_ok=True)  # Ensure test folder exists
        
        # List of directories to clean
        directories_to_clean = [
            'runs',
            'dataset',
            'static/uploads',
            'static/videos',
            'static/output',
            'static/verification',
            'annotations',
            'static/test',  # Test folder where images/videos are stored
            'static/test/crops',  # Cropped images from test folder
            'static/annotated',  # Annotated results
            'static/results',  # Test results including Excel files
            'uploads/videos',
            'uploads/images',
            'uploads/test',
            'static/downloads'
        ]
        
        # List of files to delete
        files_to_delete = [
            'labels.txt',
            'data.yaml',
            'detection_results.xlsx'
        ]
        
        # Clean directories
        for directory in directories_to_clean:
            dir_path = os.path.abspath(directory)
            if os.path.exists(dir_path):
                print(f"Deleting directory: {dir_path}")
                try:
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path, exist_ok=True)  # Recreate empty directory
                    print(f"Successfully cleaned {dir_path}")
                except Exception as e:
                    print(f"Error cleaning {dir_path}: {str(e)}")
        
        # Delete individual files
        for file in files_to_delete:
            file_path = os.path.abspath(file)
            if os.path.exists(file_path):
                print(f"Deleting file: {file_path}")
                try:
                    os.remove(file_path)
                    print(f"Successfully deleted {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        
        print("Project deletion completed")
        return jsonify({
            'success': True,
            'message': 'Project data deleted successfully'
        })
        
    except Exception as e:
        error_msg = f"Error deleting project data: {str(e)}"
        print(f"Error in delete_project: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/retraining_progress')
def retraining_progress():
    try:
        start_time = time()  # Start timing
        
        dataset_root = os.path.abspath(DATASET_ROOT)
        yaml_path = os.path.join(dataset_root, 'data.yaml')
        
        # Load the existing model for retraining
        existing_model_path = os.path.join(MODEL_DIR, 'weights/last.pt')
        if not os.path.exists(existing_model_path):
            return jsonify({'error': 'No existing model found for retraining'}), 404
        
        model = YOLO(existing_model_path)
        results = model.train(
            data=yaml_path,
            epochs=100,
            project='runs/detect',
            name='retrain',
            exist_ok=True
        )
        
        training_time = time() - start_time  # Calculate training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        # Save the retrained model in a different directory
        retrain_dir = 'runs/detect/retrain'
        model_path = os.path.join(retrain_dir, 'weights/last.pt')

        print(f"Retraining completed in {hours}h {minutes}m {seconds}s")
        
        if os.path.exists(model_path):
            print(f"Retrained model saved at: {model_path}")
            return jsonify({
                'success': True, 
                'message': f'Retraining completed in {hours}h {minutes}m {seconds}s', 
                'model_path': model_path,
                'training_time': {
                    'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds,
                    'total_seconds': training_time
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Model file not found'}), 500
        
    except Exception as e:
        print(f"Error in retraining_progress: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Add with other configuration constants
RESULTS_FOLDER = 'static/results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Add this new route to get the train image count
@app.route('/get_train_image_count')
def get_train_image_count():
    try:
        train_folder = os.path.join(DATASET_ROOT, 'train', 'images')
        if not os.path.exists(train_folder):
            return jsonify({'count': 0})
            
        # Count all image files in the train folder
        image_count = len([f for f in os.listdir(train_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        return jsonify({'count': image_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this new route to get the test file count
@app.route('/get_test_count')
def get_test_count():
    try:
        # Count images in TEST_FOLDER
        image_count = 0
        video_frame_count = 0
        
        for filename in os.listdir(TEST_FOLDER):
            file_path = os.path.join(TEST_FOLDER, filename)
            
            # Check if it's an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_count += 1
                
            # Check if it's a video
            elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
                try:
                    cap = cv2.VideoCapture(file_path)
                    if cap.isOpened():
                        video_frame_count += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                except Exception as e:
                    print(f"Error counting video frames for {filename}: {str(e)}")
        
        print(f"Found {image_count} images and {video_frame_count} video frames")  # Debug print
        
        return jsonify({
            'imageCount': image_count,
            'videoFrameCount': video_frame_count
        })
        
    except Exception as e:
        print(f"Error in get_test_count: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500


def save_cropped_images(predictions, image, output_dir):
    """Save cropped images based on predictions."""
    # Use RESULTS_FOLDER instead of static/crops
    output_dir = os.path.join(RESULTS_FOLDER, 'crops')
    os.makedirs(output_dir, exist_ok=True)

    CONFIDENCE_THRESHOLD = 0.6
    
    for i, pred in enumerate(predictions):
        boxes = pred.boxes
        for box in boxes:
            confidence = float(box.conf)
            if confidence >= CONFIDENCE_THRESHOLD:
                bbox = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = map(int, bbox)
                crop_img = image[y1:y2, x1:x2]
                crop_filename = os.path.join(output_dir, f"crop_{i}_{x1}_{y1}.jpg")
                cv2.imwrite(crop_filename, crop_img)

@app.route('/download_screenshots')
def download_screenshots():
    try:
        # Create a BytesIO object to store the zip file
        memory_file = io.BytesIO()
        
        # Create a ZipFile object
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Walk through the results directory instead of crops
            results_dir = os.path.join('static', 'results')  # Changed from 'static/crops'
            
            if not os.path.exists(results_dir):
                return jsonify({'error': 'No screenshots available'}), 404
                
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        # Add file to zip with its relative path
                        arcname = os.path.relpath(file_path, results_dir)
                        zf.write(file_path, arcname)
        
        # Seek to the beginning of the BytesIO object
        memory_file.seek(0)
        
        # Get current timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'screenshots_{timestamp}.zip'
        )
    
    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_class_list')
def get_class_list():
    try:
        classes = []
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Found classes: {classes}")  # Debug print
        else:
            print("Labels file does not exist")
            # Create the file if it doesn't exist
            with open(LABELS_FILE, 'w') as f:
                f.write('')
            
        return jsonify({
            'success': True, 
            'classes': classes,
            'file_exists': os.path.exists(LABELS_FILE)
        })
    except Exception as e:
        print(f"Error in get_class_list: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_class', methods=['POST'])
def delete_class():
    try:
        data = request.json
        class_name = data.get('class_name')
        
        if not class_name:
            return jsonify({'success': False, 'error': 'No class name provided'}), 400
            
        # Read existing classes
        classes = []
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        
        # Remove the class if it exists
        if class_name in classes:
            classes.remove(class_name)
            
            # Write updated classes back to file
            with open(LABELS_FILE, 'w') as f:
                f.write('\n'.join(classes))
            
            # Update global class_labels
            global class_labels
            class_labels = OrderedDict((label, idx) for idx, label in enumerate(classes))
            
            return jsonify({'success': True, 'message': f'Class {class_name} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Class not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/update_class_number', methods=['POST'])
def update_class_number():
    try:
        data = request.json
        class_name = data.get('class_name')
        new_number = data.get('new_number')
        
        if not class_name or new_number is None:
            return jsonify({'success': False, 'error': 'Missing class name or number'}), 400
            
        # Read existing classes
        classes = []
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        
        # Remove the class from its current position
        if class_name in classes:
            classes.remove(class_name)
            
            # Insert at new position
            new_number = max(0, min(new_number, len(classes)))  # Ensure number is in valid range
            classes.insert(new_number, class_name)
            
            # Write updated classes back to file
            with open(LABELS_FILE, 'w') as f:
                f.write('\n'.join(classes))
            
            # Update global class_labels
            global class_labels
            class_labels = OrderedDict((label, idx) for idx, label in enumerate(classes))
            
            return jsonify({
                'success': True, 
                'message': f'Class {class_name} moved to position {new_number}'
            })
        else:
            return jsonify({'success': False, 'error': 'Class not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    try:
        if 'pdfs[]' not in request.files:
            return jsonify({'error': 'No PDFs provided'}), 400

        pdf_files = request.files.getlist('pdfs[]')
        converted_images = []

        for pdf_file in pdf_files:
            # Save the PDF temporarily
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf_file.filename))
            pdf_file.save(pdf_path)

            # Convert PDF to images
            images = convert_pdf_to_images(pdf_path)
            converted_images.extend(images)

            # Remove the temporary PDF file
            os.remove(pdf_path)

        return jsonify({'success': True, 'files': converted_images})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_pdf_to_images(pdf_path):
    # Implement the logic to convert PDF to images
    # This could use a library like pdf2image
    images = convert_from_path(pdf_path)
    image_filenames = []

    for i, image in enumerate(images):
        image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path, 'PNG')
        image_filenames.append(image_filename)

    return image_filenames

@app.route('/upload_test_pdfs', methods=['POST'])
def upload_test_pdfs():
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        files = request.files.getlist('files[]')
        converted_files = []

        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp()
        try:
            for pdf_file in files:
                if not pdf_file.filename:
                    continue

                # Save PDF to temp directory
                temp_pdf_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
                pdf_file.save(temp_pdf_path)

                try:
                    # Convert PDF to images
                    images = convert_from_path(temp_pdf_path)
                    
                    # Save each page as an image
                    for i, image in enumerate(images):
                        # Generate unique filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        random_id = str(uuid.uuid4())[:8]
                        image_filename = f"pdf_page_{timestamp}_{random_id}_{i}.png"
                        image_path = os.path.join(TEST_FOLDER, image_filename)
                        
                        # Save the image
                        image.save(image_path, 'PNG')
                        converted_files.append(image_filename)
                        
                finally:
                    # Clean up the temporary PDF file
                    try:
                        os.remove(temp_pdf_path)
                    except Exception as e:
                        print(f"Error removing temp PDF: {e}")

        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error removing temp directory: {e}")

        return jsonify({
            'success': True,
            'message': f'Successfully converted {len(converted_files)} PDF pages',
            'files': converted_files
        })

    except Exception as e:
        print(f"Error processing PDFs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=6001)  # Disable reloader

