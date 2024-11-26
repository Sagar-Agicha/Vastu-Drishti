const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const imageUpload = document.getElementById("imageUpload");
const videoUpload = document.getElementById("videoUpload");
const output = document.getElementById("output");
const saveButton = document.getElementById("save");
const prevButton = document.getElementById("prevImage");
const nextButton = document.getElementById("nextImage");
const imageCounter = document.getElementById("imageCounter");

let images = [];
let currentImageIndex = 0;
let currentImage = null;
let isDrawing = false;
let startX, startY;
let currentAnnotation = null;
let tempAnnotation = null;

const CANVAS_CONTAINER_WIDTH = 800;
const CANVAS_CONTAINER_HEIGHT = 600;

let selectedVideoFile = null;
let uploadedVideoFilename = null;

let notificationQueue = [];
let isNotificationDisplaying = false;

function initializeApp() {
    console.log("Initializing app...");
    loadImages();
    
    const canvas = document.getElementById('canvas');
    const confirmClass = document.getElementById('confirmClass');
    const cancelAnnotation = document.getElementById('cancelAnnotation');
    const clearAnnotationsBtn = document.getElementById('clearAnnotations');
    
    console.log("Clear Annotations Button:", clearAnnotationsBtn);
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDrawing);
    canvas.addEventListener('mouseout', endDrawing);
    
    confirmClass.addEventListener('click', saveAnnotation);
    cancelAnnotation.addEventListener('click', () => {
        currentAnnotation = null;
        document.getElementById('classPopup').style.display = 'none';
        drawAnnotations();
    });
    
    // Image upload handler
    document.getElementById('imageUpload').addEventListener('change', uploadImages);
    
    // Navigation buttons
    document.getElementById('prevImage').addEventListener('click', showPreviousImage);
    document.getElementById('nextImage').addEventListener('click', showNextImage);
    
    // Clear buttons with debug logs
    document.getElementById('clearImages').addEventListener('click', () => {
        console.log("Clear Images clicked");
        clearImages();
    });
    
    if (clearAnnotationsBtn) {
        clearAnnotationsBtn.addEventListener('click', () => {
            console.log("Clear Annotations clicked");
            clearAllAnnotations();
        });
    } else {
        console.error("Clear Annotations button not found!");
    }
    
    // Training buttons
    document.getElementById('startProcess').addEventListener('click', startProcess);

    // Add video details button listener
    document.getElementById('showDetailsBtn').addEventListener('click', () => {
        console.log("Show Details clicked");
        if (uploadedVideoFilename) {
            showVideoDetails(uploadedVideoFilename);
        } else {
            alert('Please upload a video first');
        }
    });

    // Test section controls
    document.getElementById('testImage').addEventListener('change', uploadTestFiles);
    document.getElementById('testVideo').addEventListener('change', uploadTestFiles);
    document.getElementById('startTesting').addEventListener('click', startTesting);

    // Add download results button listener
    document.getElementById('downloadResults').addEventListener('click', downloadResults);

    // Add this to your existing initializeApp function
    document.getElementById('deleteProject').addEventListener('click', deleteProject);

    // Add these event listeners in your initializeApp function
    document.getElementById('processVideoBtn').addEventListener('click', async function() {
        if (!uploadedVideoFilename) {
            alert('Please upload a video first');
            return;
        }

        const intervalValue = document.getElementById('frameCount').value;
        
        try {
            // Disable buttons during processing
            this.disabled = true;
            document.getElementById('processVideoByCountBtn').disabled = true;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

            const response = await fetch('/process_video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: uploadedVideoFilename,
                    interval: parseInt(intervalValue)
                })
            });

            const data = await response.json();
            if (data.success) {
                showNotification(data.message, 'success');
                window.location.reload(); // Reload to show new frames
            } else {
                throw new Error(data.error || 'Failed to process video');
            }
        } catch (error) {
            console.error('Error:', error);
            showNotification('Error processing video: ' + error.message, 'error');
        } finally {
            // Re-enable buttons
            this.disabled = false;
            document.getElementById('processVideoByCountBtn').disabled = false;
            this.innerHTML = 'Extract by Time Interval';
        }
    });

    document.getElementById('processVideoByCountBtn').addEventListener('click', async function() {
        if (!uploadedVideoFilename) {
            alert('Please upload a video first');
            return;
        }

        const frameCount = document.getElementById('frameCount').value;
        
        try {
            // Disable buttons during processing
            this.disabled = true;
            document.getElementById('processVideoBtn').disabled = true;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

            const response = await fetch('/process_video_by_count', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: uploadedVideoFilename,
                    total_frames: parseInt(frameCount)
                })
            });

            const data = await response.json();
            if (data.success) {
                alert(data.message);
                window.location.reload();
            } else {
                throw new Error(data.error || 'Failed to process video');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing video: ' + error.message);
        } finally {
            // Re-enable buttons
            this.disabled = false;
            document.getElementById('processVideoBtn').disabled = false;
            this.innerHTML = 'Extract by Frame Count';
        }
    });

    document.getElementById('videoUpload').addEventListener('change', async function(e) {
        const videoUploadControls = document.getElementById('videoUploadControls');
        if (e.target.files.length > 0) {
            selectedVideoFile = e.target.files[0];
            videoUploadControls.style.display = 'block';
    
            // Show loading indicator
            const uploadLabel = document.querySelector('.upload-label[for="videoUpload"]');
            uploadLabel.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    
            try {
                const formData = new FormData();
                formData.append('video', selectedVideoFile);
    
                const response = await fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                });
    
                const data = await response.json();
                if (data.success) {
                    uploadedVideoFilename = data.filename;
                    console.log("Video uploaded:", uploadedVideoFilename);
                } else {
                    throw new Error(data.error || 'Upload failed');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error uploading video: ' + error.message);
            } finally {
                // Hide loading indicator
                uploadLabel.innerHTML = '<i class="icon"></i> Video Upload';
            }
        } else {
            selectedVideoFile = null;
            uploadedVideoFilename = null;
            videoUploadControls.style.display = 'none';
        }
    });
}

// Add this helper function to generate a unique 6-digit number
function generateUniqueId() {
    return Math.floor(100000 + Math.random() * 900000); // Generates number between 100000-999999
}

// Update the image upload function
async function uploadImages(e) {
    try {
        const fileList = e.target?.files || e;
        if (!fileList || fileList.length === 0) {
            throw new Error('No files selected');
        }

        console.log('Starting image upload...');
        const formData = new FormData();
        
        for (let file of fileList) {
            // Generate unique number and add to filename
            const uniqueId = generateUniqueId();
            const extension = file.name.split('.').pop();
            const newFileName = `${uniqueId}_${file.name}`;
            
            // Create new File object with modified name
            const modifiedFile = new File([file], newFileName, { type: file.type });
            formData.append('file', modifiedFile);
            console.log('Adding file to upload:', newFileName);
        }

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', response.status);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Upload failed');
        }

        const data = await response.json();
        console.log('Upload result:', data);

        if (data.success) {
            showNotification('Files uploaded successfully', 'success');
            await loadImages(); // Refresh the image display
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Upload failed: ' + error.message, 'error');
    }
}

// Make sure the event listener is properly set up
document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    if (imageUpload) {
        imageUpload.addEventListener('change', uploadImages);
    }
});

async function loadImages() {
    try {
        const response = await fetch('/get_images');
        const data = await response.json();
        images = data.images;
        updateImageCounter();
        if (images.length > 0) {
            displayImage(currentImageIndex);
        }
    } catch (error) {
        console.error('Error loading images:', error);
    }
}

function updateImageCounter() {
    const counter = document.getElementById('imageCounter');
    counter.style.color = 'black';
    counter.textContent = images.length > 0 
        ? `Image ${currentImageIndex + 1} of ${images.length}`
        : 'No images';
}

function setupEventListeners() {
    const canvas = document.getElementById('canvas');
    const classPopup = document.getElementById('classPopup');
    const confirmClass = document.getElementById('confirmClass');
    const cancelAnnotation = document.getElementById('cancelAnnotation');
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDrawing);
    canvas.addEventListener('mouseout', endDrawing);
    
    confirmClass.addEventListener('click', () => {
        const className = document.getElementById('className').value.trim();
        if (className) {
            saveAnnotation(className);
            classPopup.style.display = 'none';
        }
    });
    
    cancelAnnotation.addEventListener('click', () => {
        tempAnnotation = null;
        classPopup.style.display = 'none';
        redrawCanvas();
    });
}

function startDrawing(e) {
    console.log("Start drawing");
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = (e.clientX - rect.left) / canvas.width;
    startY = (e.clientY - rect.top) / canvas.height;
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const currentX = (e.clientX - rect.left) / canvas.width;
    const currentY = (e.clientY - rect.top) / canvas.height;

    // Clear and redraw everything
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw the image
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }

    // 2. Draw existing annotations
    if (images[currentImageIndex] && images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(ann => {
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            const x = (ann.x - ann.width/2) * canvas.width;
            const y = (ann.y - ann.height/2) * canvas.height;
            const w = ann.width * canvas.width;
            const h = ann.height * canvas.height;
            
            ctx.beginPath();
            ctx.rect(x, y, w, h);
            ctx.stroke();

            // Draw label
            const label = ann.class;
            ctx.font = '12px Arial';
            ctx.fillStyle = '#00ff00';
            ctx.fillText(label, x, y - 5);
        });
    }

    // 3. Draw current rectangle being drawn
    ctx.strokeStyle = '#ff0000';  // Red color for current drawing
    ctx.lineWidth = 2;
    const width = currentX - startX;
    const height = currentY - startY;
    ctx.beginPath();
    ctx.rect(
        startX * canvas.width,
        startY * canvas.height,
        width * canvas.width,
        height * canvas.height
    );
    ctx.stroke();
}

function endDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;

    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) / canvas.width;
    const endY = (e.clientY - rect.top) / canvas.height;

    // Calculate dimensions
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);

    // Only create annotation if the box has some size
    if (width > 0.01 && height > 0.01) {
        currentAnnotation = {
            x: x,
            y: y,
            width: width,
            height: height
        };

        // Show class input popup
        const popup = document.getElementById('classPopup');
        popup.style.display = 'block';
        document.getElementById('className').value = '';
        document.getElementById('className').focus();
    }
}

function showClassPopup() {
    const popup = document.getElementById('classPopup');
    const className = document.getElementById('className');
    updateClassList(); // Update the list of existing classes
    popup.style.display = 'flex';
    className.value = '';
    className.focus();
}

async function saveAnnotation(className) {
    try {
        const className = document.getElementById('className').value;
        if (!className || !currentAnnotation) {
            console.log("Missing class name or annotation");
            return;
        }

        // Convert to YOLO format (center coordinates)
        const annotation = {
            class: className,
            x: currentAnnotation.x + (currentAnnotation.width / 2),
            y: currentAnnotation.y + (currentAnnotation.height / 2),
            width: currentAnnotation.width,
            height: currentAnnotation.height
        };

        if (!images[currentImageIndex].annotations) {
            images[currentImageIndex].annotations = [];
        }

        images[currentImageIndex].annotations.push(annotation);
        
        // Save annotations to file
        saveAnnotationsToFile();
        
        // Clear current annotation and hide popup
        document.getElementById('classPopup').style.display = 'none';
        currentAnnotation = null;
        
        // Redraw canvas and update UI
        drawAnnotations();
        updateAnnotationsList();
        
        console.log("Annotation saved:", annotation); // Debug log
    } catch (error) {
        console.error('Error in saveAnnotation:', error);
        alert('Error saving annotation: ' + error.message);
    }
}

function updateAnnotationsList() {
    const tbody = document.getElementById('annotationsList');
    tbody.innerHTML = '';
    
    const annotations = images[currentImageIndex].annotations || [];
    annotations.forEach((ann, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td> <!-- Display unique number -->
            <td>${ann.class}</td>
            <td>x: ${ann.x.toFixed(3)}, y: ${ann.y.toFixed(3)}, w: ${ann.width.toFixed(3)}, h: ${ann.height.toFixed(3)}</td>
            <td>
                <button class="action-btn edit-btn" onclick="editAnnotation(${index})">Edit</button>
                <button class="action-btn delete-btn" onclick="deleteAnnotation(${index})">Delete</button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function deleteAnnotation(index) {
    images[currentImageIndex].annotations.splice(index, 1);
    updateAnnotationsList();
    drawAnnotations();
}

function editAnnotation(index) {
    const annotation = images[currentImageIndex].annotations[index];
    document.getElementById('className').value = annotation.class;
    document.getElementById('classPopup').style.display = 'block';
    
    // Store the index being edited
    tempAnnotation = annotation;
    images[currentImageIndex].annotations.splice(index, 1);
    redrawCanvas();
}

function redrawCanvas() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas and redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }
    
    // Redraw all annotations
    if (images[currentImageIndex].annotations) {
        drawAnnotations(images[currentImageIndex].annotations);
    }
}

function drawBox(ctx, box, className = '') {
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    
    let x, y, width, height;
    
    if (box.normalized) {
        width = box.width * canvasWidth;
        height = box.height * canvasHeight;
        x = (box.x * canvasWidth) - (width / 2);
        y = (box.y * canvasHeight) - (height / 2);
    } else {
        x = box.x;
        y = box.y;
        width = box.width;
        height = box.height;
    }

    // Draw rectangle with gradient stroke
    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#2196F3';
    ctx.setLineDash([]);
    ctx.rect(x, y, width, height);
    ctx.stroke();

    // Draw class name background
    if (className) {
        ctx.font = '14px Inter';
        const textWidth = ctx.measureText(className).width;
        const textHeight = 20;
        const padding = 8;
        
        ctx.fillStyle = '#2196F3';
        ctx.fillRect(x - 1, y - textHeight - padding, textWidth + padding * 2, textHeight + padding);
        
        // Draw class name text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(className, x + padding, y - padding);
    }

    // Draw corner markers
    const markerSize = 8;
    ctx.fillStyle = '#2196F3';
    
    // Top-left
    ctx.fillRect(x - markerSize/2, y - markerSize/2, markerSize, markerSize);
    // Top-right
    ctx.fillRect(x + width - markerSize/2, y - markerSize/2, markerSize, markerSize);
    // Bottom-left
    ctx.fillRect(x - markerSize/2, y + height - markerSize/2, markerSize, markerSize);
    // Bottom-right
    ctx.fillRect(x + width - markerSize/2, y + height - markerSize/2, markerSize, markerSize);
}

function drawAnnotations() {
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }

    // Draw all annotations
    if (images[currentImageIndex] && images[currentImageIndex].annotations) {
        images[currentImageIndex].annotations.forEach(ann => {
            // Draw box
            ctx.strokeStyle = '#00ff00';  // Green color
            ctx.lineWidth = 2;
            
            // Convert from center to corner coordinates for drawing
            const x = (ann.x - ann.width/2) * canvas.width;
            const y = (ann.y - ann.height/2) * canvas.height;
            const w = ann.width * canvas.width;
            const h = ann.height * canvas.height;
            
            // Draw rectangle
            ctx.beginPath();
            ctx.rect(x, y, w, h);
            ctx.stroke();

            // Draw label background
            const label = ann.class;
            ctx.font = '12px Arial';
            const textWidth = ctx.measureText(label).width;
            const padding = 2;
            
            ctx.fillStyle = '#00ff00';
            ctx.fillRect(x, y - 20, textWidth + (padding * 2), 20);

            // Draw label text
            ctx.fillStyle = '#000000';
            ctx.fillText(label, x + padding, y - 5);
        });
    }
}

async function displayImage(index) {
    if (images.length === 0 || index < 0 || index >= images.length) return;
    
    const annotations = await loadAnnotationsForImage(images[index].name);
    images[index].annotations = annotations;
    
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvas-container');
    
    const img = new Image();
    img.onload = function() {
        currentImage = img;
        
        // Determine if image is horizontal or vertical
        const isHorizontal = img.width > img.height;
        
        if (isHorizontal) {
            canvas.width = 900;
            canvas.height = 560;
            container.style.width = '900px';
            container.style.height = '560px';
        } else {
            canvas.width = 560;
            canvas.height = 900;
            container.style.width = '560px';
            container.style.height = '900px';
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const scale = Math.min(
            (canvas.width * 0.95) / img.width,
            (canvas.height * 0.95) / img.height
        );
        
        const x = (canvas.width - img.width * scale) / 2;
        const y = (canvas.height - img.height * scale) / 2;
        
        ctx.drawImage(
            img,
            x, y,
            img.width * scale,
            img.height * scale
        );
        
        drawAnnotations();
        updateAnnotationsList();
    };
    img.src = images[index].url;
    currentImageIndex = index;
    updateImageCounter();
}

function showNextImage() {
    if (currentImageIndex < images.length - 1) {
        saveAnnotationsToFile(); // Save current annotations before moving
        displayImage(currentImageIndex + 1);
        updateAnnotationsList(); // Update the list for new image
    }
}

function showPreviousImage() {
    if (currentImageIndex > 0) {
        saveAnnotationsToFile(); // Save current annotations before moving
        displayImage(currentImageIndex - 1);
        updateAnnotationsList(); // Update the list for new image
    }
}

function saveAnnotation() {
    try {
        const className = document.getElementById('className').value;
        if (!className || !currentAnnotation) {
            console.log("Missing class name or annotation");
            return;
        }

        // Convert to YOLO format (center coordinates)
        const annotation = {
            class: className,
            x: currentAnnotation.x + (currentAnnotation.width / 2),
            y: currentAnnotation.y + (currentAnnotation.height / 2),
            width: currentAnnotation.width,
            height: currentAnnotation.height
        };

        if (!images[currentImageIndex].annotations) {
            images[currentImageIndex].annotations = [];
        }

        images[currentImageIndex].annotations.push(annotation);
        
        // Save annotations to file
        saveAnnotationsToFile();
        
        // Clear current annotation and hide popup
        document.getElementById('classPopup').style.display = 'none';
        currentAnnotation = null;
        
        // Redraw canvas and update UI
        drawAnnotations();
        updateAnnotationsList();
        
        console.log("Annotation saved:", annotation); // Debug log
    } catch (error) {
        console.error('Error in saveAnnotation:', error);
        alert('Error saving annotation: ' + error.message);
    }
}

async function saveAnnotationsToFile() {
    try {
        const currentImage = images[currentImageIndex];
        if (!currentImage) {
            throw new Error('No image selected');
        }

        // Create data object for server
        const data = {
            image_file: currentImage.name,
            annotations: currentImage.annotations || []
        };

        console.log('Sending annotation data:', data);  // Debug log

        const response = await fetch('/save_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Failed to save annotations');
        }

        console.log('Annotations saved successfully:', result);
        return result;

    } catch (error) {
        console.error('Error saving annotations:', error);
        throw new Error('Failed to save annotations: ' + error.message);
    }
}

// Add this CSS to your style.css
const style = document.createElement('style');
style.textContent = `
    #canvas-container {
        position: relative;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
    #canvas {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
`;
document.head.appendChild(style);

// Your existing functions (loadImages, uploadImages, etc.) remain the same...

// Add this to your existing JavaScript
function setupVideoUpload() {
    const videoUpload = document.getElementById('videoUpload');
    videoUpload.addEventListener('change', function(e) {
        const videoUploadControls = document.getElementById('videoUploadControls');
        if (e.target.files.length > 0) {
            selectedVideoFile = e.target.files[0];
            videoUploadControls.style.display = 'block';
            uploadVideo(selectedVideoFile);
        } else {
            selectedVideoFile = null;
            uploadedVideoFilename = null;
            videoUploadControls.style.display = 'none';
        }
    });

    // Add event listener for Show Details button
    document.getElementById('showDetailsBtn').addEventListener('click', async function() {
        if (uploadedVideoFilename) {
            try {
                const response = await fetch('/get_video_details', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        filename: uploadedVideoFilename
                    })
                });

                const result = await response.json();
                if (result.success) {
                    const detailsHtml = `
                        <div class="video-details">
                            <p><strong style="color: #2196F3;" >Duration:</strong> ${result.duration.toFixed(2)} seconds</p>
                            <p><strong style="color: #2196F3;" >FPS:</strong> ${result.fps.toFixed(2)}</p>
                            <p><strong style="color: #2196F3;" >Total Frames:</strong> ${result.total_frames}</p>
                        </div>
                    `;
                    document.getElementById('videoDetails').innerHTML = detailsHtml;
                    document.getElementById('videoDetails').style.display = 'block';
                } else {
                    alert('Error getting video details: ' + result.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error getting video details');
            }
        } else {
            alert('Please upload a video first');
        }
    });
}

// Update the video upload function
async function uploadVideo(file) {
    try {
        const formData = new FormData();
        formData.append('video', file);
        
        console.log('Uploading video:', file.name);
        
        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log('Upload response:', result);
        
        if (result.success) {
            uploadedVideoFilename = result.filename;  // Store the filename with unique ID
            showNotification('Video uploaded successfully!', 'success');
            document.getElementById('videoUploadControls').style.display = 'block';
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        showNotification('Error uploading video: ' + error.message, 'error');
    }
}

async function loadAnnotationsForImage(imageFileName) {
    try {
        const response = await fetch(`/get_annotations/${imageFileName}`);
        if (response.ok) {
            const data = await response.json();
            return data.annotations;
        }
    } catch (error) {
        console.error('Error loading annotations:', error);
    }
    return [];
}

function calculateTrainingTime(imageCount) {
    const timePerImage = 69; // seconds
    return timePerImage * imageCount;
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    let timeString = '';
    if (hours > 0) timeString += `${hours}h `;
    if (minutes > 0) timeString += `${minutes}m `;
    timeString += `${remainingSeconds}s`;
    
    return timeString.trim();
}

function updateProgressBar(startTime, totalTime) {
    const currentTime = new Date().getTime();
    const elapsedTime = (currentTime - startTime) / 1000; // Convert to seconds
    const progress = Math.min((elapsedTime / totalTime) * 100, 100);
    
    const progressBar = document.querySelector('.progress-bar');
    const progressText = document.querySelector('.progress-text');
    const timeRemaining = document.getElementById('timeRemaining');
    
    progressBar.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}%`;
    
    if (progress < 100) {
        const remaining = totalTime - elapsedTime;
        timeRemaining.textContent = formatTime(remaining);
        setTimeout(() => updateProgressBar(startTime, totalTime), 1000);
    } else {
        timeRemaining.textContent = 'Complete!';
    }
}

async function startProcess() {
    try {
        showNotification('Starting training process...', 'info');
        
        // First prepare the training data
        const prepareResponse = await fetch('/prepare_training', {
            method: 'POST'
        });
        
        if (!prepareResponse.ok) {
            throw new Error('Failed to prepare training data');
        }
        
        const prepareResult = await prepareResponse.json();
        if (!prepareResult.success) {
            throw new Error(prepareResult.error || 'Failed to prepare training data');
        }
        
        showNotification(prepareResult.message, 'success');
        
        // Get the count of images in the train folder
        const response = await fetch('/get_train_image_count');
        const data = await response.json();
        const imageCount = data.count;
        
        // Calculate total training time
        const totalTime = calculateTrainingTime(imageCount);
        
        // Show training progress
        document.getElementById('trainingProgress').style.display = 'block';
        
        // Start progress bar
        const startTime = new Date().getTime();
        const progressPromise = updateTrainingProgressBar(startTime, totalTime);
        
        // Start the training process in parallel
        const trainingPromise = fetch('/training_progress').then(response => response.json());
        
        // Wait for both progress and training to complete
        const [trainingResult] = await Promise.all([
            trainingPromise,
            progressPromise
        ]);
        
        if (trainingResult.success) {
            showNotification('Training completed successfully!', 'success');
        } else {
            throw new Error(trainingResult.error || 'Training failed');
        }
        
    } catch (error) {
        console.error('Error in process:', error);
        showNotification('Error: ' + error.message, 'error');
    }
}

function updateTrainingProgressBar(startTime, totalTime) {
    return new Promise((resolve) => {
        function update() {
            const currentTime = new Date().getTime();
            const elapsedTime = (currentTime - startTime) / 1000;
            const progress = Math.min((elapsedTime / totalTime) * 100, 100);
            
            const progressBar = document.querySelector('#trainingProgress .progress-bar');
            const progressText = document.querySelector('#trainingProgress .progress-text');
            const timeRemaining = document.getElementById('timeRemaining');
            
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${Math.round(progress)}%`;
            
            if (progress < 100) {
                const remaining = totalTime - elapsedTime;
                timeRemaining.textContent = formatTime(remaining);
                setTimeout(update, 1000);
            } else {
                timeRemaining.textContent = 'Complete!';
                resolve();
            }
        }
        
        update();
    });
}

async function clearAnnotations() {
    if (confirm('Are you sure you want to clear ALL annotations for ALL images? This cannot be undone.')) {
        try {
            // Show loading state
            const clearBtn = document.querySelector('.clear-btn');
            const originalText = clearBtn.innerHTML;
            clearBtn.innerHTML = 'Clearing...';
            clearBtn.disabled = true;

            // Clear annotations
            images.forEach(image => {
                image.annotations = [];
            });

            // Clear display
            drawAnnotations();
            updateAnnotationsList();

            // Clear files
            await clearAllAnnotationFiles();

            // Show success message
            showNotification('All annotations cleared successfully', 'success');

        } catch (error) {
            console.error('Error clearing annotations:', error);
            showNotification('Failed to clear annotations', 'error');
        } finally {
            // Reset button
            const clearBtn = document.querySelector('.clear-btn');
            clearBtn.innerHTML = originalText;
            clearBtn.disabled = false;
        }
    }
}

function showNotification(message, type = 'info') {
    // Add notification to queue
    notificationQueue.push({ message, type });
    
    // If no notification is currently showing, display the next one
    if (!isNotificationDisplaying) {
        displayNextNotification();
    }
}

function displayNextNotification() {
    if (notificationQueue.length === 0) {
        isNotificationDisplaying = false;
        return;
    }

    isNotificationDisplaying = true;
    const { message, type } = notificationQueue.shift();

    // Remove any existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Add close button
    const closeButton = document.createElement('span');
    closeButton.className = 'notification-close';
    closeButton.innerHTML = '×';
    closeButton.onclick = () => {
        notification.remove();
        isNotificationDisplaying = false;
        displayNextNotification();
    };

    // Add message
    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;

    notification.appendChild(messageSpan);
    notification.appendChild(closeButton);
    document.body.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
            isNotificationDisplaying = false;
            displayNextNotification();
        }
    }, 3000);
}

async function clearAllAnnotationFiles() {
    try {
        const response = await fetch('/clear_all_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error('Failed to clear annotations');
        }

        const result = await response.json();
        if (result.success) {
            console.log('All annotations cleared successfully');
        }
    } catch (error) {
        console.error('Error clearing annotations:', error);
        throw error;
    }
}

// Add this function to populate existing classes
function updateClassList() {
    const datalist = document.getElementById('existingClasses');
    const existingClasses = new Set();
    
    // Collect all existing class names
    images.forEach(img => {
        if (img.annotations) {
            img.annotations.forEach(ann => {
                if (ann.class) {
                    existingClasses.add(ann.class);
                }
            });
        }
    });
    
    // Update datalist
    datalist.innerHTML = '';
    existingClasses.forEach(className => {
        const option = document.createElement('option');
        option.value = className;
        datalist.appendChild(option);
    });
}

// Add keyboard handling for the popup
document.getElementById('className').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        document.getElementById('confirmClass').click();
    }
});

// Close popup when clicking outside
document.getElementById('classPopup').addEventListener('click', function(e) {
    if (e.target === this) {
        document.getElementById('cancelAnnotation').click();
    }
});

function updateAnnotationsList() {
    const tbody = document.getElementById('annotationsList');
    tbody.innerHTML = ''; // Clear existing list
    
    if (!images[currentImageIndex] || !images[currentImageIndex].annotations) {
        return;
    }
    
    images[currentImageIndex].annotations.forEach((ann, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td> <!-- Display unique number -->
            <td>${ann.class}</td>
            <td>
                x: ${ann.x.toFixed(3)}<br>
                y: ${ann.y.toFixed(3)}<br>
                w: ${ann.width.toFixed(3)}<br>
                h: ${ann.height.toFixed(3)}
            </td>
            <td>
                <button class="action-btn edit-btn" onclick="editAnnotation(${index})">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="action-btn delete-btn" onclick="deleteAnnotation(${index})">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });

    // Update image counter
    updateImageCounter();
}

// Update the button style to make it more prominent for a dangerous action
document.querySelector('.clear-btn').style.cssText = `
    background-color: #dc3545;
    color: white;
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.3s ease;
`;

document.getElementById('processVideoBtn').addEventListener('click', async function() {
    if (!uploadedVideoFilename) {
        alert('Please upload a video first');
        return;
    }

    console.log('Processing video:', uploadedVideoFilename);
    const intervalValue = document.getElementById('frameCount').value;
    
    try {
        this.disabled = true;
        document.getElementById('processVideoByCountBtn').disabled = true;
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

        const response = await fetch('/process_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: uploadedVideoFilename,
                interval: parseInt(intervalValue)
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to process video');
        }

        const data = await response.json();
        if (data.success) {
            showNotification(data.message, 'success');
            window.location.reload();
        } else {
            throw new Error(data.error || 'Failed to process video');
        }
    } catch (error) {
        console.error('Error processing video:', error);
        showNotification('Error processing video: ' + error.message, 'error');
    } finally {
        this.disabled = false;
        document.getElementById('processVideoByCountBtn').disabled = false;
        this.innerHTML = 'Extract by Time Interval';
    }
});

function clearImages() {
    console.log("clearImages function called");
    if (confirm('Are you sure you want to delete all images and their annotations? This cannot be undone.')) {
        console.log("User confirmed image deletion");
        fetch('/clear_images', {
            method: 'POST',
        })
        .then(response => {
            console.log("Clear images response:", response);
            return response.json();
        })
        .then(data => {
            console.log("Clear images data:", data);
            if (data.success) {
                return fetch('/clear_all_annotations', {
                    method: 'POST'
                });
            }
            throw new Error(data.error || 'Failed to clear images');
        })
        .then(response => response.json())
        .then(data => {
            console.log("Clear annotations data:", data);
            if (data.success) {
                // Reset the UI
                images = [];
                currentImageIndex = 0;
                currentImage = null;
                
                // Clear canvas
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Clear annotations list
                const annotationsList = document.getElementById('annotationsList');
                if (annotationsList) {
                    annotationsList.innerHTML = '';
                }
                
                // Update UI elements
                updateImageCounter();
                
                alert('Images and annotations cleared successfully');
                loadImages();
            }
        })
        .catch(error => {
            console.error('Error in clearImages:', error);
            alert('Error clearing images and annotations');
        });
    } else {
        console.log("User cancelled image deletion");
    }
}

// Update the video processing to delete the video after extraction
async function processVideo(filename, frameCount) {
    try {
        const totalFrames = document.getElementById('totalFrames').value;
        const response = await fetch('/process_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: filename,
                frame_interval: frameCount,
                total_frames: totalFrames
            })
        });

        const result = await response.json();
        if (result.success) {
            showNotification('Video processed successfully', 'success');
            return result;
        } else {
            throw new Error(result.error || 'Failed to process video');
        }
    } catch (error) {
        console.error('Error processing video:', error);
        showNotification('Error processing video', 'error');
        throw error;
    }
}

// Add the clearAllAnnotations function
function clearAllAnnotations() {
    console.log("clearAllAnnotations function called");
    if (confirm('Are you sure you want to delete all annotations? This cannot be undone.')) {
        console.log("User confirmed deletion");
        fetch('/clear_all_annotations', {
            method: 'POST'
        })
        .then(response => {
            console.log("Response received:", response);
            return response.json();
        })
        .then(data => {
            console.log("Data received:", data);
            if (data.success) {
                console.log("Clearing annotations from UI");
                // Clear annotations from current display
                if (images[currentImageIndex]) {
                    images[currentImageIndex].annotations = [];
                }
                
                // Clear annotations list
                const annotationsList = document.getElementById('annotationsList');
                if (annotationsList) {
                    annotationsList.innerHTML = '';
                }
                
                // Redraw canvas
                drawAnnotations();
                
                alert('All annotations cleared successfully');
            } else {
                throw new Error(data.error || 'Failed to clear annotations');
            }
        })
        .catch(error => {
            console.error('Error in clearAllAnnotations:', error);
            alert('Error clearing annotations');
        });
    } else {
        console.log("User cancelled deletion");
    }
}

// Add the showVideoDetails function
async function showVideoDetails(filename) {
    try {
        console.log("Fetching video details for:", filename);
        const response = await fetch('/get_video_details', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename: filename })
        });

        const data = await response.json();
        console.log("Video details received:", data);

        if (data.success) {
            const detailsDiv = document.getElementById('videoDetails');
            detailsDiv.innerHTML = `
                <div style="margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
                    <p style="color: black;"><strong>FPS:</strong> ${data.fps.toFixed(2)}</p>
                    <p style="color: black;"><strong>Total Frames:</strong> ${data.total_frames}</p>
                    <p style="color: black;"><strong>Duration:</strong> ${data.duration.toFixed(2)} seconds</p>
                </div>
            `;
            detailsDiv.style.display = 'block';
        } else {
            throw new Error(data.error || 'Failed to get video details');
        }
    } catch (error) {
        console.error('Error getting video details:', error);
        alert('Error getting video details: ' + error.message);
    }
}

async function uploadTestFiles(e) {
    let files = Array.from(e.target.files);
    if (!files || files.length === 0) return;

    console.log("Starting test file upload...");
    
    // Create FormData with let instead of const
    let formData = new FormData();
    
    try {
        // Process files in sequence to avoid file handle conflicts
        for (const file of files) {
            try {
                const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'video/mp4', 'application/pdf'];
                if (!validTypes.includes(file.type)) {
                    throw new Error(`Invalid file type: ${file.type}. Only images, videos, and PDFs are allowed.`);
                }
                
                // Create unique filename
                const timestamp = Date.now();
                const random = Math.floor(Math.random() * 10000);
                const extension = file.name.split('.').pop();
                const newFileName = `test_${timestamp}_${random}.${extension}`;
                
                // Read file data and create new blob
                const buffer = await file.arrayBuffer();
                const blob = new Blob([buffer], { type: file.type });
                const newFile = new File([blob], newFileName, { type: file.type });
                
                // Add to FormData
                formData.append('files[]', newFile);
                console.log("Added file to upload:", newFileName, "Type:", file.type);
                
                // Small delay between files
                await new Promise(resolve => setTimeout(resolve, 100));
            } catch (fileError) {
                console.error('Error processing file:', file.name, fileError);
                showNotification(`Error processing ${file.name}: ${fileError.message}`, 'error');
            }
        }

        // Ensure all file handles are released before upload
        await new Promise(resolve => setTimeout(resolve, 500));

        showNotification('Uploading test files...', 'info');
        
        const response = await fetch('/upload_test_files', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error (${response.status})`);
        }

        const result = await response.json();
        console.log("Upload result:", result);

        if (result.success) {
            showNotification(`Successfully uploaded ${result.files.length} files`, 'success');
            await updateTestFileCount();
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('Error uploading test files:', error);
        showNotification('Error uploading files: ' + error.message, 'error');
    } finally {
        try {
            // Reset file input
            e.target.value = '';
            
            // Clear references
            formData = null;
            files = null;
            
            // Force garbage collection
            await new Promise(resolve => setTimeout(resolve, 100));
            
            if (window.gc) {
                window.gc();
            }
        } catch (cleanupError) {
            console.warn('Non-critical cleanup error:', cleanupError);
        }
    }
}

function startTesting() {
    // Prevent multiple calls
    if (this.dataset.processing === 'true') {
        console.log('Testing already in progress');
        return;
    }
    
    this.dataset.processing = 'true';
    const mode = document.getElementById('modeSwitch').checked ? 'screenshot' : 'ocr';
    const downloadBtn = document.getElementById('downloadResults');
    
    // Disable download button initially and during testing
    if (downloadBtn) {
        downloadBtn.disabled = false;
    }
    
    console.log('Starting testing in mode:', mode);
    showNotification(`Starting testing in ${mode} mode...`, 'info');
    
    fetch('/start_testing', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode: mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Testing completed in ${mode} mode:`, data);
            if (mode === 'ocr') {
                // Enable download button for OCR results
                const downloadBtn = document.getElementById('downloadResults');
                if (downloadBtn) {
                    downloadBtn.disabled = false;
                    downloadBtn.onclick = downloadOCRResults;
                }
                showNotification('OCR testing completed successfully!', 'success');
            } else {
                showNotification('Screenshots saved successfully!', 'success');
                // Enable download button for screenshots
                const downloadBtn = document.getElementById('downloadResults');
                if (downloadBtn) {
                    downloadBtn.disabled = false;
                    downloadBtn.onclick = downloadScreenshots;
                }
            }
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error during testing: ' + error.message, 'error');
    });
}

function downloadOCRResults() {
    showNotification('Preparing OCR results for download...', 'info');
    
    window.location.href = '/download_results';

}

function downloadScreenshots() {
    showNotification('Preparing screenshots for download...', 'info');
    
    fetch('/download_screenshots', {
        method: 'GET',
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'screenshots.zip';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        showNotification('Screenshots downloaded successfully!', 'success');
    })
    .catch(error => {
        console.error('Error downloading screenshots:', error);
        showNotification('Error downloading screenshots: ' + error.message, 'error');
    });
}

function displayOCRResults(results) {
    // Create or get a container for OCR results
    let resultsContainer = document.getElementById('ocrResults');
    if (!resultsContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.id = 'ocrResults';
        document.querySelector('.test-content').appendChild(resultsContainer);
    }

    // Clear previous results
    resultsContainer.innerHTML = '<h3>OCR Results:</h3>';

    // Display results for each file
    for (const [filename, classResults] of Object.entries(results)) {
        const fileSection = document.createElement('div');
        fileSection.className = 'file-section';
        
        const fileHeader = document.createElement('h4');
        fileHeader.textContent = filename;
        fileSection.appendChild(fileHeader);

        // Display results for each class
        for (const [className, texts] of Object.entries(classResults)) {
            const classDiv = document.createElement('div');
            classDiv.className = 'class-results';
            
            const classHeader = document.createElement('h5');
            classHeader.textContent = className;
            classDiv.appendChild(classHeader);

            const textList = document.createElement('ul');
            texts.forEach(text => {
                const li = document.createElement('li');
                li.textContent = text;
                textList.appendChild(li);
            });
            
            classDiv.appendChild(textList);
            fileSection.appendChild(classDiv);
        }

        resultsContainer.appendChild(fileSection);
    }
}

function toggleMode() {
    const switchElement = document.getElementById('modeSwitch');
    const mode = switchElement.checked ? 'screenshot' : 'ocr';
    const downloadBtn = document.getElementById('downloadResults');
    
    // Debug information
    console.log('Toggle mode:', mode);
    
    // Update UI elements based on mode
    const screenshotLabel = document.querySelector('.screenshot-label');
    const ocrLabel = document.querySelector('.ocr-label');
    
    if (mode === 'screenshot') {
        screenshotLabel.classList.add('active-mode');
        ocrLabel.classList.remove('active-mode');
        // Clear any existing OCR results
        const ocrResults = document.getElementById('ocrResults');
        if (ocrResults) {
            ocrResults.style.display = 'none';
        }
    } else {
        screenshotLabel.classList.remove('active-mode');
        ocrLabel.classList.add('active-mode');
    }
    
    // Reset download button state
    if (downloadBtn) {
        downloadBtn.disabled = true;
    }

    // Store the current mode in localStorage
    localStorage.setItem('testMode', mode);
}

// Add this event listener for the Extract by Frame Count button
document.getElementById('processVideoByCountBtn').addEventListener('click', async function() {
    if (!uploadedVideoFilename) {
        alert('Please upload a video first');
        return;
    }

    // Disable both buttons and show processing state
    const timeIntervalBtn = document.getElementById('processVideoBtn');
    const frameCountBtn = document.getElementById('processVideoByCountBtn');
    
    timeIntervalBtn.disabled = true;
    frameCountBtn.disabled = true;
    frameCountBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    const totalFrames = document.getElementById('totalFrames').value;
    
    try {
        // First, clear all annotations
        await video_annotation_delete();
        console.log('Annotations cleared');

        const response = await fetch('/process_video_by_count', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: uploadedVideoFilename,
                total_frames: parseInt(totalFrames)
            })
        });

        const data = await response.json();
        if (data.success) {
            // Show success message
            alert(data.message);
            
            // Refresh the page
            window.location.reload();
        } else {
            throw new Error(data.error || 'Failed to process video');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing video: ' + error.message);
        
        // Reset buttons state in case of error
        timeIntervalBtn.disabled = false;
        frameCountBtn.disabled = false;
        frameCountBtn.innerHTML = 'Extract by Frame Count';
    }
});

// Add this function to clear annotations
async function clearAnnotations() {
    try {
        const response = await fetch('/clear_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to clear annotations');
        }
    } catch (error) {
        console.error('Error clearing annotations:', error);
    }
}

// Add this new function to delete video annotations
async function video_annotation_delete() {
    try {
        const response = await fetch('/clear_all_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to clear annotations');
        }
        
        console.log('Annotations cleared successfully');
    } catch (error) {
        console.error('Error clearing annotations:', error);
    }
}

// Add these functions to your existing JavaScript

// Function to toggle class manager visibility
function toggleClassManager() {
    const panel = document.getElementById('classManagerPanel');
    const isVisible = panel.style.display === 'block';
    panel.style.display = isVisible ? 'none' : 'block';
    if (!isVisible) {
        refreshClassList();
    }
}

function downloadResults() {
    fetch('/download_results')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'test_results.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        })
        .catch(error => {
            console.error('Error downloading results:', error);
            showNotification('Error downloading results: ' + error.message, 'error');
        });
}

// Function to refresh the class list
async function refreshClassList() {
    try {
        const response = await fetch('/get_class_list');
        const data = await response.json();
        
        console.log("Class list response:", data); // Debug log
        
        const classList = document.getElementById('classList');
        if (!classList) {
            console.error('classList element not found');
            return;
        }
        
        if (!data.success) {
            console.error('Error loading classes:', data.error);
            classList.innerHTML = `
                <div class="class-item" style="justify-content: center; color: #dc2626;">
                    <i class="fas fa-exclamation-circle"></i>&nbsp; Error loading classes
                </div>`;
            return;
        }
        
        if (!data.classes || data.classes.length === 0) {
            classList.innerHTML = `
                <div class="class-item" style="justify-content: center; color: #94a3b8;">
                    <i class="fas fa-info-circle"></i>&nbsp; No classes defined
                </div>`;
            return;
        }
        
        classList.innerHTML = '';
        data.classes.forEach((className, index) => {
            const classItem = document.createElement('div');
            classItem.className = 'class-item';
            classItem.innerHTML = `
                <span class="class-number">${index}</span>
                <span class="class-name">${className}</span>
                <button class="delete-btn" onclick="deleteClass('${className}')">
                    <i class="fas fa-trash"></i>
                </button>
            `;
            classList.appendChild(classItem);
        });
    } catch (error) {
        console.error('Error loading classes:', error);
        const classList = document.getElementById('classList');
        if (classList) {
            classList.innerHTML = `
                <div class="class-item" style="justify-content: center; color: #dc2626;">
                    <i class="fas fa-exclamation-circle"></i>&nbsp; Error loading classes
                </div>`;
        }
    }
}

// Function to delete a class
async function deleteClass(className) {
    if (!confirm(`Are you sure you want to delete the class "${className}"?`)) {
        return;
    }
    
    try {
        const response = await fetch('/delete_class', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ class_name: className })
        });
        
        const data = await response.json();
        if (data.success) {
            window.refreshClassList();
        } else {
            alert('Error deleting class: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error deleting class');
    }
}

// Add new function to handle class number updates
async function updateClassNumber(className, newNumber, inputElement) {
    try {
        const response = await fetch('/update_class_number', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                class_name: className,
                new_number: parseInt(newNumber)
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification(`Updated number for class "${className}"`, 'success');
            await refreshClassList(); // Refresh to show updated order
        } else {
            showNotification('Error updating class number: ' + data.error, 'error');
            inputElement.value = inputElement.getAttribute('data-original-index');
            await refreshClassList(); // Refresh to restore original order
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error updating class number', 'error');
        inputElement.value = inputElement.getAttribute('data-original-index');
        await refreshClassList(); // Refresh to restore original order
    }
}

// Add event listeners
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('classManagerModal');
    const closeBtn = document.getElementById('closeClassManager');
    
    // Create the button with updated styling
    const classManagerBtn = document.createElement('button');
    classManagerBtn.innerHTML = '<i class="fas fa-tags"></i> Manage Classes';
    classManagerBtn.className = 'manage-classes-btn';
    
    // Add the button to the document body
    document.body.appendChild(classManagerBtn);

    // Function to toggle class manager visibility
    function toggleClassManager() {
        if (!modal) {
            console.error('Class manager modal not found');
            return;
        }
        modal.style.display = modal.style.display === 'block' ? 'none' : 'block';
        if (modal.style.display === 'block') {
            refreshClassList();
        }
    }

    // Add click event to the button
    classManagerBtn.addEventListener('click', toggleClassManager);

    // Close modal when clicking the close button
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
        });
    }

    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Function to refresh the class list
    async function refreshClassList() {
        try {
            const response = await fetch('/get_class_list');
            const data = await response.json();
            
            const classList = document.getElementById('classList');
            if (!classList) {
                console.error('Class list element not found');
                return;
            }
            
            classList.innerHTML = ''; // Clear existing list
            
            if (!data.classes || data.classes.length === 0) {
                classList.innerHTML = `
                    <div class="class-item" style="justify-content: center; color: #9ca3af;">
                        <i class="fas fa-info-circle"></i>&nbsp; No classes defined
                    </div>`;
                return;
            }
            
            data.classes.forEach(className => {
                const classItem = document.createElement('div');
                classItem.className = 'class-item';
                classItem.innerHTML = `
                    <span class="class-name">${className}</span>
                    <button class="delete-btn" onclick="deleteClass('${className}')">
                        <i class="fas fa-trash"></i>
                    </button>
                `;
                classList.appendChild(classItem);
            });
        } catch (error) {
            console.error('Error loading classes:', error);
            const classList = document.getElementById('classList');
            if (classList) {
                classList.innerHTML = `
                    <div class="class-item" style="justify-content: center; color: #dc2626;">
                        <i class="fas fa-exclamation-circle"></i>&nbsp; Error loading classes
                    </div>`;
            }
        }
    }
    // Make functions globally available
    window.toggleClassManager = toggleClassManager;
    window.refreshClassList = refreshClassList;

    // Add search functionality
    const searchInput = document.getElementById('classSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const classItems = document.querySelectorAll('.class-item');
            
            classItems.forEach(item => {
                const className = item.querySelector('.class-name').textContent.toLowerCase();
                if (className.includes(searchTerm)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
});

// Add this function for deleting the project
async function deleteProject() {
    if (!confirm('Are you sure you want to delete all project data? This action cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch('/delete_project', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();
        
        if (data.success) {
            // Show success notification
            showNotification('Project deleted successfully', 'success');
            
            // Reset UI state
            images = [];
            currentImageIndex = 0;
            currentImage = null;
            
            // Clear canvas
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Clear annotations list
            const annotationsList = document.getElementById('annotationsList');
            if (annotationsList) {
                annotationsList.innerHTML = '';
            }
            
            // Update UI elements
            updateImageCounter();
            loadImages();
            
            // Reload the page after a short delay
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error deleting project:', error);
        showNotification('Error deleting project', 'error');
    }
}

// Add this helper function for notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Make sure to add this to your DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
    const deleteProjectBtn = document.getElementById('deleteProject');
    if (deleteProjectBtn) {
        deleteProjectBtn.addEventListener('click', deleteProject);
    }
});

// Add this function to handle PDF uploads
async function uploadPDFs(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    console.log("Starting PDF upload...");

    const formData = new FormData();
    for (let file of files) {
        formData.append('pdfs[]', file);
        console.log("Adding PDF to upload:", file.name);
    }

    try {
        showNotification('Uploading PDFs...', 'info');

        const response = await fetch('/upload_pdfs', {
            method: 'POST',
            body: formData
        });

        console.log("Upload response status:", response.status);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Upload result:", result);

        showNotification(`Successfully uploaded ${result.files.length} PDFs`, 'success');

        // Reset the file input
        e.target.value = '';
        // Reload the page after a short delay
        setTimeout(() => {
            window.location.reload();
        }, 1500);

    } catch (error) {
        console.error('Error uploading PDFs:', error);
        showNotification('Error uploading PDFs: ' + error.message, 'error');
    }
}

// Add event listeners for the new PDF upload buttons
document.addEventListener('DOMContentLoaded', function() {
    const pdfUpload = document.getElementById('pdfUpload');
    const testPdf = document.getElementById('testPdf');

    if (pdfUpload) {
        pdfUpload.addEventListener('change', uploadPDFs);
        console.log("PDF upload listener added");
    }

    if (testPdf) {
        testPdf.addEventListener('change', uploadTestFiles);
        console.log("Test PDF upload listener added");
    }
});


async function uploadTestPDF(e) {
    let files = Array.from(e.target.files);
    if (!files || files.length === 0) return;

    console.log("Starting PDF upload...");
    let formData = new FormData();

    try {
        // Process PDFs sequentially
        for (const file of files) {
            if (file.type !== 'application/pdf') {
                showNotification(`Invalid file type: ${file.type}. Only PDFs are allowed.`, 'error');
                continue;
            }

            // Create unique filename for PDF
            const timestamp = Date.now();
            const random = Math.floor(Math.random() * 10000);
            const newFileName = `test_pdf_${timestamp}_${random}.pdf`;

            // Create a new blob from the PDF
            const buffer = await file.arrayBuffer();
            const blob = new Blob([buffer], { type: 'application/pdf' });
            const newFile = new File([blob], newFileName, { type: 'application/pdf' });

            formData.append('files[]', newFile);
            console.log("Added PDF to upload:", newFileName);

            // Add delay between processing files
            await new Promise(resolve => setTimeout(resolve, 200));
        }

        showNotification('Processing PDFs...', 'info');

        const response = await fetch('/upload_test_pdfs', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error (${response.status})`);
        }

        const result = await response.json();
        console.log("Upload result:", result);

        if (result.success) {
            showNotification(`Successfully processed ${result.files.length} PDF pages`, 'success');
            await updateTestFileCount();
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('Error uploading PDFs:', error);
        showNotification('Error uploading PDFs: ' + error.message, 'error');
    } finally {
        // Clean up
        e.target.value = '';
        formData = null;
        files = null;
    }
}


// Separate function for handling image/video uploads
async function uploadTestMedia(e) {
    let files = Array.from(e.target.files);
    if (!files || files.length === 0) return;

    console.log("Starting media upload...");
    let formData = new FormData();

    try {
        for (const file of files) {
            const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'video/mp4'];
            if (!validTypes.includes(file.type)) {
                showNotification(`Invalid file type: ${file.type}. Only images and videos are allowed.`, 'error');
                continue;
            }

            const timestamp = Date.now();
            const random = Math.floor(Math.random() * 10000);
            const extension = file.name.split('.').pop();
            const newFileName = `test_media_${timestamp}_${random}.${extension}`;

            const buffer = await file.arrayBuffer();
            const blob = new Blob([buffer], { type: file.type });
            const newFile = new File([blob], newFileName, { type: file.type });

            formData.append('files[]', newFile);
            console.log("Added media file to upload:", newFileName);

            await new Promise(resolve => setTimeout(resolve, 200));
        }

        showNotification('Uploading media files...', 'info');

        const response = await fetch('/upload_test_files', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error (${response.status})`);
        }

        const result = await response.json();
        if (result.success) {
            showNotification(`Successfully uploaded ${result.files.length} files`, 'success');
            await updateTestFileCount();
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('Error uploading media:', error);
        showNotification('Error uploading media: ' + error.message, 'error');
    } finally {
        e.target.value = '';
        formData = null;
        files = null;
    }
}

async function updateTestFileCount() {
    try {
        console.log("Fetching test file count...");
        const response = await fetch('/get_test_file_count');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("Test file count response:", data);
        
        if (data.success) {
            // Update UI elements with the new counts
            const counts = data.counts || { images: 0, videos: 0 };
            
            // Update count displays
            document.getElementById('testImageCount').textContent = counts.images || 0;
            document.getElementById('testVideoCount').textContent = counts.videos || 0;
            
            // Enable/disable the start testing button based on total count
            const totalCount = (counts.images || 0) + (counts.videos || 0);
            const startTestingBtn = document.getElementById('startTesting');
            if (startTestingBtn) {
                startTestingBtn.disabled = totalCount === 0;
            }
            
            console.log("File counts updated successfully");
            return true;
        } else {
            console.error('Error getting test file count:', data.error);
            return false;
        }
    } catch (error) {
        console.error('Error updating test file count:', error);
        return false;
    }
}

// Call this function after file uploads and when the page loads
document.addEventListener('DOMContentLoaded', function() {
    updateTestFileCount();
});
// Update the event listeners
document.addEventListener('DOMContentLoaded', function() {
    // PDF upload listener
    const testPdfInput = document.getElementById('testPdf');
    if (testPdfInput) {
        testPdfInput.removeEventListener('change', uploadTestFiles); // Remove old listener
        testPdfInput.addEventListener('change', uploadTestPDF);
        console.log("PDF upload listener added");
    }

    // Media upload listener
    const testMediaInput = document.getElementById('testImage');
    if (testMediaInput) {
        testMediaInput.removeEventListener('change', uploadTestFiles); // Remove old listener
        testMediaInput.addEventListener('change', uploadTestMedia);
        console.log("Media upload listener added");
    }
});

