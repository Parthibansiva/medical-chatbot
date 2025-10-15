from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import whisper
import subprocess
import os
from tempfile import NamedTemporaryFile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")
logger.info("Loading Whisper model... (this may take a minute)")
model = whisper.load_model("base")
logger.info("Whisper model loaded successfully ✅")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

def convert_to_wav(input_file, output_file):
    subprocess.run([
        "ffmpeg", "-i", input_file, "-ar", "16000", "-ac", "1", output_file
    ], check=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


model = whisper.load_model("base")

def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path
    ], check=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ Updated transcription endpoint using open-source Whisper
@app.post("/transcribe_audio")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Save uploaded file
        with NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio_path = temp_audio.name

        # Convert to wav
        wav_path = temp_audio_path.replace(".webm", ".wav")
        subprocess.run(["ffmpeg", "-y", "-i", temp_audio_path, "-ar", "16000", "-ac", "1", wav_path], check=True)

        # Transcribe
        result = model.transcribe(wav_path)
        transcript = result["text"]

        # Cleanup
        os.remove(temp_audio_path)
        os.remove(wav_path)

        return {"transcript": transcript}

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Transcription failed: {str(e)}"}

    
@app.post("/upload_and_query")
async def upload_and_query(
    query: str = Form(...),
    image: UploadFile = File(None)  # <-- image is optional now
):
    try:
        messages = [{"role": "user", "content": []}]

        # Always include the text query
        messages[0]["content"].append({"type": "text", "text": query})

        # If image was uploaded, add it
        if image:
            image_content = await image.read()
            if not image_content:
                raise HTTPException(status_code=400, detail="Empty image file uploaded.")

            # Validate and encode image
            try:
                img = Image.open(io.BytesIO(image_content))
                img.verify()
            except Exception as e:
                logger.error(f"Invalid image format: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

            encoded_image = base64.b64encode(image_content).decode("utf-8")
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })

        def make_api_request(model):
            response = requests.post(
                GROQ_API_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            return response

        # Make requests to both models
        llama_response = make_api_request("meta-llama/llama-4-scout-17b-16e-instruct")
        llava_response = make_api_request("meta-llama/llama-4-maverick-17b-128e-instruct")

        # Process responses
        responses = {}
        for model, response in [("llama", llama_response), ("llava", llava_response)]:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Processed response from {model} API: {answer[:100]}...")
                responses[model] = answer
            else:
                logger.error(f"Error from {model} API: {response.status_code} - {response.text}")
                responses[model] = f"Error from {model} API: {response.status_code}"

        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)