import onnxruntime as ort
import numpy as np
import cv2  # OpenCV for image processing

# --- Configuration ---
ONNX_MODEL_PATH = 'resources/template_val_sai.onnx'  # Path to your new .onnx model
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- Load the ONNX Model (once, when the server starts) ---
try:
    print("Loading template validation ONNX model...")
    # Create an inference session with the ONNX model
    # The providers list specifies the execution backend. 'CPUExecutionProvider' is a safe default.
    # You can add 'CUDAExecutionProvider' for GPU acceleration if available.
    template_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # Get the input and output names of the model
    input_name = template_session.get_inputs()[0].name
    output_name = template_session.get_outputs()[0].name
    
    print("Template ONNX model loaded successfully.")
    print(f" - Model Input Name: '{input_name}'")
    print(f" - Model Output Name: '{output_name}'")
    
except Exception as e:
    print(f"FATAL ERROR: Could not load template ONNX model from '{ONNX_MODEL_PATH}'.")
    print(f"   Error details: {e}")
    template_session = None

# --- Main Function to be called by your application (e.g., FastAPI) ---

def run_template_check(image_np: np.ndarray) -> dict:
    """
    Takes an image as a NumPy array and returns a template validation score using ONNX Runtime.

    Args:
        image_np: The image read by OpenCV (in BGR format).

    Returns:
        A dictionary with the confidence score, e.g., {"score": 0.95}
    """
    if template_session is None:
        # Fallback if the model failed to load
        return {"score": 0.0, "details": "Model not loaded"}

    try:
        # 1. Preprocess the image (must be IDENTICAL to training preprocessing)
        
        # Convert from BGR (OpenCV's default) to RGB
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Resize image to the model's expected input size
        # Use INTER_AREA for shrinking and INTER_CUBIC for enlarging for best results
        resized_image = cv2.resize(rgb_image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Expand dimensions to create a batch of 1: (H, W, C) -> (1, H, W, C)
        img_array_expanded = np.expand_dims(resized_image, axis=0)
        
        # Replicate tf.keras.applications.efficientnet.preprocess_input
        # This function scales pixel values from [0, 255] to [-1, 1].
        # It's crucial to also convert the data type to float32 for the model.
        processed_img = (img_array_expanded.astype(np.float32) / 127.5) - 1.0

        # 2. Make prediction using ONNX Runtime
        # The input must be a dictionary mapping input_name to the data.
        # The output is a list of arrays, one for each model output.
        prediction_scores = template_session.run([output_name], {input_name: processed_img})
        
        # The raw sigmoid output is the confidence score.
        # It's a value between 0.0 and 1.0.
        # The result is inside a list and an array, so we extract it.
        genuine_score = float(prediction_scores[0][0][0])

        # 3. Return the score in the required format
        return {
            "score": genuine_score,
            "details": "Template validation complete"
        }

    except Exception as e:
        print(f"ERROR in ONNX template validation: {e}")
        return {"score": 0.0, "details": str(e)}

# --- Example Usage (to test the function) ---
if __name__ == '__main__':
    # This block will only run when you execute this script directly.
    # It demonstrates how to use the function.
    if template_session:
        # Create a dummy blank image for testing purposes
        # In a real scenario, you would load an image:
        # test_image = cv2.imread("path/to/your/test_image.jpg")
        print("\n--- Running a test prediction on a dummy black image ---")
        dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)

        # Run the check
        result = run_template_check(dummy_image)

        # Print the result
        print(f"Prediction Result: {result}")