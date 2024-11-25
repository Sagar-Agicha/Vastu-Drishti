import cv2
import face_recognition

# Load all reference images of the person and create encodings
reference_images = ["Testing_Images/test1.jpg", "Testing_Images/test2.jpg", "Testing_Images/test3.jpg", "Testing_Images/test4.jpg", "Testing_Images/test5.jpg"]
reference_encodings = []

for image_path in reference_images:
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:  # Make sure the image contains a face
        reference_encodings.append(encodings[0])

# Open the video file
video = cv2.VideoCapture("Testing_Videos/test1.mp4")
frame_rate = video.get(cv2.CAP_PROP_FPS)
frame_interval = int(frame_rate)  # Process one frame per second

# Prepare to write the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated_video.mp4', fourcc, frame_rate, (int(video.get(3)), int(video.get(4))))

duration_in_frames = 0

batch_size = 16  # Number of frames to process in a batch
frames = []
frame_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    frames.append(frame)
    frame_count += 1

    # Process in batches
    if len(frames) == batch_size or not success:
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        all_face_locations = face_recognition.batch_face_locations(rgb_frames)
        all_face_encodings = [face_recognition.face_encodings(f, loc) for f, loc in zip(rgb_frames, all_face_locations)]

        for frame, face_locations, face_encodings in zip(frames, all_face_locations, all_face_encodings):
            for face_location, encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(reference_encodings, encoding)
                top, right, bottom, left = face_location

                if True in matches:
                    duration_in_frames += 1
                    # Draw a green rectangle around recognized faces
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    # Draw a red rectangle around unrecognized faces
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Write the frame with annotations
            out.write(frame)

        frames = []  # Clear the batch

video.release()
out.release()

# Calculate time in seconds
duration_in_seconds = duration_in_frames / frame_rate
duration_in_minutes = duration_in_seconds / 60

print(f"The person was in the video for {duration_in_minutes:.2f} minutes.")
