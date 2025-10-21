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
from tempfile import NamedTemporaryFile
from googletrans import Translator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load Whisper base model (good for Tamil and English)
logger.info("Loading Whisper base model... (this may take a minute)")
model = whisper.load_model("base")
logger.info("Whisper base model loaded successfully ✅")

# Initialize translator
translator = Translator()
logger.info("Google Translator initialized ✅")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Updated transcription endpoint with Tamil to English translation
@app.post("/transcribe_audio")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("en")
):
    try:
        logger.info(f"Transcribing audio - Selected language: {language}")
        
        # Save uploaded file
        with NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio_path = temp_audio.name

        # Convert to wav format (Whisper works best with 16kHz mono)
        wav_path = temp_audio_path.replace(".webm", ".wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_audio_path, 
                "-ar", "16000", "-ac", "1", wav_path
            ], check=True, capture_output=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion error: {e.stderr.decode()}")
            raise HTTPException(status_code=500, detail="Audio conversion failed")

        # Always use the selected language for transcription
        transcribe_options = {
            "fp16": False,
            "language": language
        }
        
        result = model.transcribe(wav_path, **transcribe_options)
        transcript = result["text"].strip()

        logger.info(f"✅ Transcription successful")
        logger.info(f"   Transcript: {transcript[:100]}...")

        # Translate to English if needed
        translated_text = transcript
        translation_performed = False
        
        if language != "en":
            try:
                translated_text = translator.translate(transcript, src=language, dest="en").text
                translation_performed = True
            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                translated_text = transcript
                translation_performed = False

        # Cleanup temporary files
        try:
            os.remove(temp_audio_path)
            os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Could not delete temp files: {e}")

        return {
            "transcript": translated_text,
            "original_transcript": transcript,
            "detected_language": language,
            "requested_language": language,
            "translated": translation_performed
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {str(e)}")
        return {"error": f"Audio conversion failed. Please check FFmpeg installation."}
    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return {"error": f"Transcription failed: {str(e)}"}

    
@app.post("/upload_and_query")
async def upload_and_query(
    query: str = Form(...),
    image: UploadFile = File(None)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)