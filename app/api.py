from fastapi import APIRouter, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os
import google.generativeai as genai

# NOTE: load_dotenv() is removed here as it is handled by main.py

# --- Configuration ---
# os.getenv will now find the key loaded by main.py
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("INFO: Gemini API configured successfully.")
    except Exception as e:
        print(f"CRITICAL: Could not configure Gemini API. Error: {e}")
else:
    print("WARNING: No Gemini API key found. Add GEMINI_API_KEY to your .env file.")

# --- Router ---
router = APIRouter()

# --- Load Model ---
try:
    MODEL_PATH = "skin_disease_model_mobilenetv2_final.keras"
    # Ensure the model file is accessible from the root directory when running uvicorn
    model = tf.keras.models.load_model(MODEL_PATH) 
    class_names = ['Eczema', 'Normal_Skin', 'Pigmentation', 'Rosacea', 'Scar', 'Vitiligo', 'melanoma']
    print(f"INFO: Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load model '{MODEL_PATH}'. Error: {e}")
    model = None


# --- Gemini API Helper ---
def get_gemini_response(prediction, confidence):
    # NOTE: Changing the message to correctly reference the .env file (if it still fails)
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Please add your key to the .env file."

    try:
        # Using a safer model name and assuming 'gemini-pro' is available
        model_gemini = genai.GenerativeModel("gemini-2.5-flash") 
        
        prompt = f"""
        Act as a supportive AI skin health assistant.
        My AI image model predicted the skin condition as '{prediction}' with {confidence*100:.0f}% confidence.

        Please write:
        1. A simple explanation of what '{prediction}' means in plain language.
        2. A safe recommended next step.
        3. A disclaimer: this is NOT a medical diagnosis and they should consult a doctor.
        """
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"ERROR: Could not get response from Gemini: {e}")
        return "Could not retrieve additional information at this time. Please consult a medical professional."


# --- Prediction Endpoint ---
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0  # ✅ normalize
    image_batch = np.expand_dims(image_array, 0)


    # Prediction
    predictions = model.predict(image_batch, verbose=0)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(np.max(predictions))

    # Gemini smart response
    helpful_info = get_gemini_response(predicted_class_name, confidence)

    return {
        "prediction": predicted_class_name,
        "confidence": confidence,
        "helpful_info": helpful_info
    }



# from fastapi import APIRouter, UploadFile, File, HTTPException, Form
# from typing import Optional
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io, os
# import google.generativeai as genai

# # NOTE: load_dotenv() is removed here as it is handled by main.py

# # --- Configuration ---
# # os.getenv will now find the key loaded by main.py
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if GEMINI_API_KEY:
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         print("INFO: Gemini API configured successfully.")
#     except Exception as e:
#         print(f"CRITICAL: Could not configure Gemini API. Error: {e}")
# else:
#     print("WARNING: No Gemini API key found. Add GEMINI_API_KEY to your .env file.")

# # --- Router ---
# router = APIRouter()

# # --- Load Model ---
# try:
#     MODEL_PATH = "skin_disease_model.h5"
#     # Ensure the model file is accessible from the root directory when running uvicorn
#     model = tf.keras.models.load_model(MODEL_PATH) 
#     class_names = ['Eczema', 'Pigmentation', 'Rosacea', 'Melanoma']
#     print(f"INFO: Model '{MODEL_PATH}' loaded successfully.")
# except Exception as e:
#     print(f"CRITICAL: Failed to load model '{MODEL_PATH}'. Error: {e}")
#     model = None


# # --- Gemini API Helper ---
# def get_gemini_response(prediction, confidence, patient_info=None):
#     """
#     Get AI-generated health information from Gemini
    
#     Args:
#         prediction: The predicted skin condition
#         confidence: Confidence score of the prediction
#         patient_info: Optional dictionary with patient details (age, symptoms, etc.)
#     """
#     if not GEMINI_API_KEY:
#         return "Gemini API key not configured. Please add your key to the .env file."

#     try:
#         model_gemini = genai.GenerativeModel("gemini-2.0-flash-exp") 
        
#         # Build a more personalized prompt if patient info is available
#         additional_context = ""
#         if patient_info:
#             if patient_info.get('age'):
#                 additional_context += f"\nPatient age: {patient_info['age']}"
#             if patient_info.get('symptoms'):
#                 additional_context += f"\nReported symptoms: {patient_info['symptoms']}"
        
#         prompt = f"""
#         Act as a supportive AI skin health assistant.
#         My AI image model predicted the skin condition as '{prediction}' with {confidence*100:.0f}% confidence.
#         {additional_context}

#         Please write:
#         1. A simple explanation of what '{prediction}' means in plain language.
#         2. Common symptoms and characteristics of this condition.
#         3. Safe recommended next steps for care.
#         4. When to seek immediate medical attention.
#         5. A clear disclaimer: this is NOT a medical diagnosis and they should consult a dermatologist or healthcare provider.
        
#         Keep the response empathetic, informative, and encouraging them to seek professional medical advice.
#         """
#         response = model_gemini.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         print(f"ERROR: Could not get response from Gemini: {e}")
#         return "Could not retrieve additional information at this time. Please consult a medical professional."


# # --- Original Prediction Endpoint (keeping for backward compatibility) ---
# @router.("/predict")
# async def predict(file: UploadFile = File(...)):
#     """
#     Simple prediction endpoint - accepts only an image file
#     """
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

#     try:
#         # Read and validate image
#         contents = await file.read()
        
#         # Validate file is an image
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
#         image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
#         image_array = np.array(image) / 255.0  # normalize
#         image_batch = np.expand_dims(image_array, 0)

#         # Prediction
#         predictions = model.predict(image_batch, verbose=0)
#         predicted_class_index = np.argmax(predictions)
#         predicted_class_name = class_names[predicted_class_index]
#         confidence = float(np.max(predictions))

#         # Gemini smart response
#         helpful_info = get_gemini_response(predicted_class_name, confidence)

#         return {
#             "status": "success",
#             "prediction": predicted_class_name,
#             "confidence": confidence,
#             "all_predictions": {
#                 class_names[i]: float(predictions[0][i]) 
#                 for i in range(len(class_names))
#             },
#             "helpful_info": helpful_info
#         }
    
#     except Exception as e:
#         print(f"ERROR in prediction: {e}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# # --- Enhanced Prediction Endpoint with Form Data ---
# @router.post("/predict-with-info")
# async def predict_with_patient_info(
#     file: UploadFile = File(...),
#     name: Optional[str] = Form(None),
#     age: Optional[int] = Form(None),
#     symptoms: Optional[str] = Form(None),
#     medical_history: Optional[str] = Form(None)
# ):
#     """
#     Enhanced prediction endpoint - accepts image + patient information
#     This allows for more personalized AI responses
#     """
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

#     try:
#         # Read and validate image
#         contents = await file.read()
        
#         # Validate file is an image
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
#         # Validate image size (e.g., max 10MB)
#         max_size = 10 * 1024 * 1024  # 10MB
#         if len(contents) > max_size:
#             raise HTTPException(status_code=400, detail="Image file too large. Maximum size is 10MB")
        
#         image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
#         image_array = np.array(image) / 255.0  # normalize
#         image_batch = np.expand_dims(image_array, 0)

#         # Prediction
#         predictions = model.predict(image_batch, verbose=0)
#         predicted_class_index = np.argmax(predictions)
#         predicted_class_name = class_names[predicted_class_index]
#         confidence = float(np.max(predictions))

#         # Prepare patient info for Gemini
#         patient_info = {
#             'age': age,
#             'symptoms': symptoms,
#             'medical_history': medical_history
#         }

#         # Get personalized Gemini response
#         helpful_info = get_gemini_response(predicted_class_name, confidence, patient_info)

#         return {
#             "status": "success",
#             "prediction": predicted_class_name,
#             "confidence": confidence,
#             "all_predictions": {
#                 class_names[i]: float(predictions[0][i]) 
#                 for i in range(len(class_names))
#             },
#             # "patient_info": {
#             #     "name": name,
#             #     "age": age,
#             #     "symptoms": symptoms
#             # },
#             # "helpful_info": helpful_info,
#             # "file_info": {
#             #     "filename": file.filename,
#             #     "content_type": file.content_type,
#             #     "size_bytes": len(contents)
#             # }
#         }
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"ERROR in prediction with info: {e}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# # --- Health Check Endpoint ---
# @router.get("/health")
# async def health_check():
#     """
#     Check if the API and model are working properly
#     """
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "gemini_configured": GEMINI_API_KEY is not None,
#         "supported_conditions": class_names if model else []
#     }