import io
import base64
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, ConnectionException, ClientException
from vipas.logger import LoggerClient
from PIL import Image

logger = LoggerClient(__name__)

def pre_process(data):
    try:
        # Preprocess the input image
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        image = image.resize((224, 224))  # Resize according to your model's requirement
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
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
    model_id = "mdl-egd1sfadhctl3"
    vps_model_client = model.ModelClient()
    response = vps_model_client.predict(model_id=model_id, input_data=input_data)
    
    # Convert the response into an output image
    if response.get("output_image"):
        output_image_data = base64.b64decode(response["output_image"])
        return Image.open(io.BytesIO(output_image_data))
    
    return None
