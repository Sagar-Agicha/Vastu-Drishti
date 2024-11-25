from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition

# Load YOLOv8 model
model = YOLO('Models/last.pt')

def detect_faces(image):
    # Detect faces in the image
    results = model(image)
    return results

def crop_faces(image, results):
    # Extract face regions from the image
    faces = []
    for result in results:
        # Assuming result is an object with a 'boxes' attribute
        # and 'boxes' is a list of bounding box coordinates
        for box in result.boxes:
            # Debugging: Print the shape and content of box.xyxy
            print("box.xyxy:", box.xyxy)
            print("Shape of box.xyxy:", box.xyxy.shape)

            # Check if box.xyxy is a 2D tensor and has at least one row
            if box.xyxy.ndim == 2 and box.xyxy.shape[0] > 0:
                # Ensure the first row has at least 4 elements
                if box.xyxy.shape[1] >= 4:
                    x1, y1, x2, y2 = box.xyxy[0][:4].int().tolist()  # Safely access the first four elements
                    face = image[y1:y2, x1:x2]
                    faces.append(face)
                else:
                    print("Warning: box.xyxy does not have enough elements.")
            else:
                print("Warning: box.xyxy is not a 2D tensor or is empty.")
    return faces

def get_embeddings(faces):
    # Get embeddings for each face using face_recognition
    embeddings = []
    for face in faces:
        # Convert the face to RGB format as face_recognition expects RGB images
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Get the face encodings (embeddings)
        face_encodings = face_recognition.face_encodings(rgb_face)
        if face_encodings:
            embeddings.append(face_encodings[0])  # Use the first encoding if available
    return embeddings

def create_known_embeddings(image_paths):
    # Create known embeddings from a list of image paths
    known_embeddings = []
    for path in image_paths:
        image = cv2.imread(path)
        results = detect_faces(image)
        faces = crop_faces(image, results)
        embeddings = get_embeddings(faces)
        known_embeddings.extend(embeddings)
    return known_embeddings

def match_faces(embeddings, known_embeddings, threshold=0.6):
    # Compare embeddings with known embeddings
    matches = []
    for embedding in embeddings:
        distances = [np.linalg.norm(embedding - known_embedding) for known_embedding in known_embeddings]
        min_distance = min(distances) if distances else float('inf')
        if min_distance < threshold:
            matches.append(True)  # Match found
        else:
            matches.append(False)  # No match
    return matches

# Example usage
image_paths = ['Testing_Images/test1.jpg', 'Testing_Images/test2.jpg', 'Testing_Images/test3.jpg', 'Testing_Images/test4.jpg', 'Testing_Images/test5.jpg']
known_embeddings = create_known_embeddings(image_paths)

# Process a video
video_path = 'Testing_Videos/test1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)  # Number of frames to skip (1 second)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'MJPG', 'X264', etc.
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 1 second
    if frame_count % frame_interval == 0:
        results = detect_faces(frame)
        faces = crop_faces(frame, results)
        embeddings = get_embeddings(faces)
        matches = match_faces(embeddings, known_embeddings)

        # Annotate the frame with detection results
        for i, (result, match) in enumerate(zip(results, matches)):
            try:
                x1, y1, x2, y2 = result.boxes.xyxy[0][:4].int().tolist()
            except AttributeError:
                print("Warning: result.boxes is not defined or does not have 'xyxy' attribute.")
                continue

            color = (0, 255, 0) if match else (0, 0, 255)  # Green for match, red for no match
            label = "Match" if match else "No Match"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Put label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame to the output video
        out.write(frame)

        # Display the annotated frame
        cv2.imshow('Video', frame)

    frame_count += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()