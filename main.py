from fastapi import FastAPI
from dotenv import load_dotenv # Import
load_dotenv()                 # Execute: This loads the .env file in the root directory

from app import api # Imports the router from your app/api.py file

# Initialize the main FastAPI app
app = FastAPI(title="Skin Disease Detection API")

# Include the router from app/api.py
# The prefix="/api" means all routes will start with /api (e.g., /api/predict)
app.include_router(api.router, prefix="/api")

# A root endpoint just to check if the server is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Disease Detection API! Go to /docs to test the API."}





# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from typing import Optional

# load_dotenv()  # Execute: This loads the .env file in the root directory

# from app import api  # Imports the router from your app/api.py file

# # Initialize the main FastAPI app
# app = FastAPI(title="Skin Disease Detection API")

# # Add CORS middleware to allow frontend requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include the router from app/api.py\
# # The prefix="/api" means all routes will start with /api (e.g., /api/predict)
# app.include_router(api.router, prefix="/api")


# # A root endpoint just to check if the server is running
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Skin Disease Detection API! Go to /docs to test the API."}


# # POST endpoint to handle form data with image upload
# @app.post("/api/")
# async def handle_form_data(
#     image: UploadFile = File(...),
#     # name: Optional[str] = Form(None),
#     # age: Optional[int] = Form(None),
#     # symptoms: Optional[str] = Form(None)
# ):
#     """
#     Handle form data from frontend including image upload
    
#     Parameters:
#     - image: The uploaded image file (required)
#     - name: Patient name (optional)
#     - age: Patient age (optional)
#     - symptoms: Description of symptoms (optional)
#     """
#     # Read the image file
#     image_data = await image.read()
    
#     # Process your data here
#     # For example, you can save the image, run predictions, etc.
    
#     return {
#         "status": "success",
#         "message": "Form data received successfully",
#         "data": {
#             "filename": image.filename,
#             "content_type": image.content_type,
#             "file_size": len(image_data),
#             # "name": name,
#             # "age": age,
#             # "symptoms": symptoms
#         }
#     }

