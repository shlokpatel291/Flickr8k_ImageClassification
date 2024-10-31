from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from .inference import generate_caption

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Captioning API. Use /generate-caption/ to generate captions."}

@app.post("/generate-caption/")
async def caption_image(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is an image
        if not file.content_type.startswith('image/'):
            return JSONResponse(content={"error": "Uploaded file is not an image."}, status_code=400)
        
        # Read the uploaded file as bytes
        image_bytes = await file.read()
        
        # Generate caption using the bytes
        caption = generate_caption(image_bytes)
        
        return JSONResponse(content={"caption": caption})

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

