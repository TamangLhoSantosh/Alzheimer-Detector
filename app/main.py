from fastapi import FastAPI, File, UploadFile, HTTPException
from app.services.detector_service import PredictionService
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Initialize the PredictionService with the path to the pre-trained model
prediction_service = PredictionService("./app/models/Alzheimer_Detector.keras")


# Helper function to map numerical predictions to their respective classification labels
def names(number):
    """Converts numerical prediction to a human-readable label for Alzheimer's stages."""
    if number == 0:
        return "Non Demented"
    elif number == 1:
        return "Mild Dementia"
    elif number == 2:
        return "Moderate Dementia"
    elif number == 3:
        return "Very Mild Dementia"
    else:
        return "Error in Prediction"  # Handles unexpected classification result


# API endpoint for predicting the Alzheimer's stage from an image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for predicting the Alzheimer's disease stage based on an uploaded image.
    Only accepts .jpg, .jpeg, and .png image formats.
    """

    # Validate the file format (only allow image files with .jpg, .jpeg, .png extensions)
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,  # HTTP status code for Bad Request
            detail="Invalid file format. Please upload a .jpg, .jpeg, or .png file.",
        )

    try:
        # Read the uploaded file content into memory
        content = await file.read()

        # Open the image file using PIL from the in-memory byte content
        img = Image.open(io.BytesIO(content))

        # Use the prediction service to get the confidence and classification (stage of Alzheimer's)
        confidence, classification = prediction_service.predict(img)

        # Return the prediction with confidence in percentage and human-readable classification label
        return {
            "confidence": f"{confidence:.2f}%",
            "prediction": names(
                classification
            ),  # Convert the classification number to a label
        }

    except Exception as e:
        # Catch any errors during image processing or prediction and raise a 500 error with the message
        raise HTTPException(
            status_code=500, detail=f"Error processing the image: {str(e)}"
        )
