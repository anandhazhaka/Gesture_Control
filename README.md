# Gesture-Controlled Image Viewer

This project demonstrates a **gesture-controlled image viewer** that leverages hand tracking and recognition to interact with images through real-time gestures. By using the **MediaPipe** library for hand tracking and **OpenCV** for image manipulation, users can zoom in/out of images and capture photos with gestures, offering an intuitive, hands-free experience.

## Key Features

### 1. **Hand Gesture Recognition:**
   The system recognizes two key hand gestures:
   - **Strict Thumb-Up Gesture:** When the user performs a strict thumbs-up gesture, the current image displayed is saved to disk.
   - **Palm Open Gesture:** The system detects when the palm is open to enable **pinch-to-zoom** functionality for zooming in and out of images.

### 2. **Zooming with Pinch Gesture:**
   - Using the **pinch gesture** between the thumb and index finger, users can dynamically zoom in and out on images. The pinch distance is detected and used to scale the image up or down, simulating the zoom-in/zoom-out effect.
   
### 3. **Image Navigation:**
   - The application loads images from a specified directory (`"radio"`), and users can browse through the images with gestures. Each image is displayed at a consistent resolution, with zoom adjustments based on hand gestures.
   
### 4. **Image Display and Information Overlay:**
   - The application displays images with a zoom scale indicator and the current image number. It allows users to track the zoom level and the sequence of images they are viewing.
   - A small thumbs-up image is overlaid on the screen when the strict thumbs-up gesture is detected.
   
### 5. **Real-Time Gesture Feedback:**
   - The system provides real-time feedback on the webcam feed, displaying messages such as "Zoom Mode" when the palm open gesture is detected or "Thumbs Up Detected - Photo Saved!" when a thumbs-up gesture is detected.

### 6. **Hands-Free Interaction:**
   - Users interact with the image gallery entirely through hand gestures, with no physical input required from a mouse or keyboard. This makes it an ideal solution for a hands-free environment.

## Requirements

To run this project, you'll need the following libraries:

- **OpenCV** (for image processing and webcam interaction)
- **MediaPipe** (for hand gesture recognition)
- **NumPy** (for numerical operations)
- **Python** (version 3.x)

You can install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy
```

## How It Works

1. **Hand Gesture Detection:**
   - The webcam feed is captured in real-time.
   - **MediaPipe** is used to detect hand landmarks and track hand movements.
   - The system checks for specific gestures (thumbs-up and palm open) to trigger actions like saving the current image or zooming in/out.

2. **Zooming with Pinch Gesture:**
   - When a right hand is detected, the distance between the thumb and index finger is calculated.
   - The pinch distance is used to adjust the zoom scale of the displayed image.

3. **Image Loading and Display:**
   - The images from the specified directory are loaded, resized to fit the screen, and displayed in sequence.
   - The current image index and zoom level are shown on the screen.

4. **Thumbs-Up Gesture for Photo Capture:**
   - A "strict" thumbs-up gesture is detected by checking if the thumb is significantly above its base joint and all other fingers are curled.
   - When detected, the current image is saved to disk as `photo_capture.jpg`.

## Usage

1. **Zooming:** Perform a pinch gesture (thumb and index finger) to zoom in and out of the current image.
2. **Image Navigation:** Browse through images by performing gestures to trigger zoom or interaction.
3. **Photo Capture:** Perform a strict thumbs-up gesture to save the current image.

To stop the application, press `q` in the window displaying the webcam feed.

## Project Structure

- **gesture_control.py**: Main Python script that runs the gesture-based image control.
- **radio/**: Directory where images are stored (ensure to add your own images here).
- **thumbs_up.png**: Image to display when a strict thumbs-up gesture is detected.
- **photo_capture.jpg**: Saved image when the thumbs-up gesture is detected.

## Future Enhancements

- **Multi-hand support:** Extend support for interactions with both hands simultaneously.
- **Swipe gestures:** Implement swipe gestures to navigate through images more intuitively.
- **Custom gestures:** Allow users to define their own gestures for different actions (e.g., next image, previous image).
