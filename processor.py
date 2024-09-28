import io
import base64
import cv2
import numpy as np
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, ConnectionException, ClientException
from vipas.logger import LoggerClient
from PIL import Image

logger = LoggerClient(__name__)

def pre_process(data):
    try:
        # Decode the base64 image
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        # Convert the image to a format suitable for OpenCV
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Resize according to your model's requirement
        image_resized = cv2.resize(image, (224, 224))
        # Convert back to PIL Image for further processing
        pil_image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        preprocessed_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info("Preprocessing completed successfully.")
        return preprocessed_data
    except Exception as err:
        logger.critical(f"Error in preprocessing: {str(err)}")
        raise

def post_process(output, threshold=0.5):
    try:
        # Process the model's output
        results = output.get('predictions', [])
        classes = ['Non-Violence', 'Violence']
        processed_output = {}
        
        for index, result in enumerate(results):
            if index < len(classes):
                confidence = result['confidence']
                if confidence >= threshold:  # Apply threshold
                    processed_output[classes[index]] = confidence

        logger.info("Postprocessing completed successfully.")
        return processed_output
    except Exception as err:
        logger.critical(f"Error in postprocessing: {str(err)}")
        raise

def predict_image(input_data):
    model_id = "mdl-egd1sfadhctl3"  # Replace with your actual model ID
    vps_model_client = model.ModelClient()
    
    try:
        response = vps_model_client.predict(model_id=model_id, input_data=input_data)
        return response  # Return the raw response to process it later
    except UnauthorizedException as e:
        logger.error("Unauthorized exception: " + str(e))
        raise
    except NotFoundException as e:
        logger.error("Not found exception: " + str(e))
        raise
    except ConnectionException as e:
        logger.error("Connection exception: " + str(e))
        raise
    except ClientException as e:
        logger.error("Client exception: " + str(e))
        raise
    except Exception as e:
        logger.error("Exception when calling model->predict: %s\n" % e)
        raise

def process_image(data, threshold=0.5):
    # Preprocess the image
    preprocessed_data = pre_process(data)
    
    # Get predictions
    response = predict_image(preprocessed_data)
    
    # Post-process the predictions
    results = post_process(response, threshold)
    
    return results
