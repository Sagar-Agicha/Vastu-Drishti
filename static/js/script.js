document.addEventListener('DOMContentLoaded', function() {
    // Remove any code that triggers the class name popup here
    loadImages();  // Keep this if you want to load images on page load
    // Other initialization code that doesn't involve the class name popup
});

// The class name popup should only be triggered when creating a new annotation
// This is typically in your annotation creation logic, something like:
function createNewAnnotation(event) {
    // Show the class name popup only when user starts drawing a new box
    const className = prompt("Enter class name:");  // or your custom popup logic
    if (className) {
        // Continue with annotation creation
    }
} 