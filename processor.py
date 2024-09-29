import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64
import cv2

def pre_process(input_video_base64):
    """
    Preprocess the input video for the model.
    
    Parameters:
        input_video_base64 (str): Base64 encoded input video.
        
    Returns:
        list: A list of preprocessed image arrays ready for model input.
        str: Base64 encoded original video.
    """
    # Decode the base64 video input
    video_data = base64.b64decode(input_video_base64)
    
    # Save the video to a temporary file
    video_file_path = '/tmp/input_video.mp4'
    with open(video_file_path, 'wb') as f:
        f.write(video_data)

    # Initialize video capture
    vidcap = cv2.VideoCapture(video_file_path)

    processed_frames = []
    original_frames = []

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        
        # Resize the frame to match model input size (128x128)
        resized_frame = cv2.resize(frame, (128, 128))
        
        # Convert the frame to a numpy array and normalize it to [0, 1]
        frame_np = resized_frame.astype(np.float32) / 255.0
        
        # Add a batch dimension for model input (shape: (1, 128, 128, 3))
        frame_np = np.expand_dims(frame_np, axis=0)

        # Store processed frame
        processed_frames.append(frame_np.tolist())

        # Store the original frame for later encoding (for video output)
        original_frames.append(frame)

    vidcap.release()

    # Convert original frames to base64 string for output
    buffered = BytesIO()
    for frame in original_frames:
        img = Image.fromarray(frame)
        img.save(buffered, format="PNG")  # Save as PNG
    original_video_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return processed_frames, original_video_base64  # Convert to list for JSON serialization


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

# Example usage:
# processed_frames, original_video_base64 = pre_process(base64_input_video)
# predictions = model.predict(processed_frames)
# results = post_process(predictions)
