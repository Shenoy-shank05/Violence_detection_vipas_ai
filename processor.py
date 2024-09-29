import numpy as np
import tensorflow as tf
from io import BytesIO
import cv2
import base64

def pre_process(input_video_base64):
    """
    Preprocess the input video for the model.
    
    Parameters:
        input_video_base64 (str): Base64 encoded input video.
        
    Returns:
        tuple: A tuple containing:
            - List: Preprocessed frames ready for model input.
            - str: Base64 encoded original video.
    """
    # Decode the base64 video input
    video_data = base64.b64decode(input_video_base64)

    # Write the video data to a temporary file
    temp_video_path = '/tmp/temp_video.mp4'
    with open(temp_video_path, 'wb') as temp_video_file:
        temp_video_file.write(video_data)

    # Initialize video capture
    cap = cv2.VideoCapture(temp_video_path)

    frames = []
    original_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to match model input size (128x128)
        frame_resized = cv2.resize(frame, (128, 128))  # Adjusted size
        frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]

        # Append the processed frame
        frames.append(frame_normalized)
        original_frames.append(frame)  # Keep original frames for output

    cap.release()
    
    # Convert processed frames to numpy array
    frames_np = np.array(frames)
    
    # Convert original frames to base64 string for output
    original_video_buffer = BytesIO()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(original_video_buffer, fourcc, 30, (original_frames[0].shape[1], original_frames[0].shape[0]))
    
    for frame in original_frames:
        out.write(frame)
    out.release()

    original_video_base64 = base64.b64encode(original_video_buffer.getvalue()).decode('utf-8')

    return frames_np.tolist(), original_video_base64  # Convert to list for JSON serialization


def post_process(predictions):
    """
    Post-process the model predictions to extract relevant information.
    
    Parameters:
        predictions (tf.Tensor): The raw output from the model.
        
    Returns:
        dict: A dictionary containing filtered detection results.
    """
    boxes = predictions['detection_boxes'].numpy()  # Get bounding box coordinates
    scores = predictions['detection_scores'].numpy()  # Get confidence scores
    classes = predictions['detection_classes'].numpy()  # Get detected classes
    num_detections = int(predictions['num_detections'].numpy())  # Get number of detections

    # Lists to hold filtered predictions
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    # Filter out predictions based on score threshold
    for i in range(num_detections):
        if scores[i] > 0.5:  # Confidence threshold
            filtered_boxes.append(boxes[i].tolist())  # Convert to list for JSON serialization
            filtered_scores.append(float(scores[i]))  # Convert to float for JSON serialization
            filtered_classes.append(int(classes[i]))  # Convert to int for JSON serialization

    # Compile filtered predictions into a dictionary
    filtered_predictions = {
        'detection_boxes': filtered_boxes,
        'detection_scores': filtered_scores,
        'detection_classes': filtered_classes,
        'num_detections': len(filtered_scores)
    }
    
    return filtered_predictions
