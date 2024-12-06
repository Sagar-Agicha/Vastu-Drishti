import cv2
from ultralytics import YOLO
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Sagar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Load the YOLOv8 model
model = YOLO("Models/last.pt")  # Replace with your model path

# Open the video file
video_path = "Testing_Videos/test1.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

with open("labels.txt", "r") as file:
    labels = file.read().splitlines()

print("Number of labels: ", len(labels))

for name in labels:
    print(name)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "annotated_video_with_labels.mp4"  # Output video file

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define font and color for text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (0, 255, 0)  # Green
thickness = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Extract detections
    for result in results[0].boxes:
        # Get coordinates and label
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        label = result.cls  # Class ID
        conf = result.conf.item()  # Confidence score

        if label == 0:
            # Get class name from the model's class list

                class_name = model.names[int(label)]
                label_text = f"{class_name} {conf:.2f}"  # e.g., "person 0.85"

                # Crop the region inside bounding box
                cropped = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                results = model(gray)

                for result in results[0].boxes:
                    label = result.cls
                    conf = result.conf.item()   

                    for name in labels:
                        if label_text == name:
                            cropped = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            # Perform OCR on the cropped region
                            ocr_result = pytesseract.image_to_string(cropped)
                            print(ocr_result)

            # # Draw the bounding box
            # cv2.rectangle(frame, (x1, y1), (x2, y2), font_color, thickness)

            # # Put the label text above the bounding box
            # text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            # text_x, text_y = x1, y1 - 10
            # if text_y < 10:  # Adjust if label goes out of frame
            #     text_y = y1 + 20

            # # Background rectangle for text
            # cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), font_color, -1)

            # # Write the label text
            # cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Write the annotated frame to the output video
    out.write(frame)

    # Optionally, display the frame (press 'q' to quit)
    cv2.imshow("Annotated Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video with labels saved to: {output_path}")
