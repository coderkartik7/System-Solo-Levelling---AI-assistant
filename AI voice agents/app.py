# Importing packages
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import Optional
import assemblyai as aai
import time
from google import genai

# Loading all the environment files from .env
load_dotenv()

#Get API credentials from env file
MURF_API_KEY = os.getenv("MURF_API_KEY")
MURF_API_URL = os.getenv("MURF_API_URL","https://api.murf.ai/v1/speech/generate")
ASS_API_KEY = os.getenv("ASSEMBLY_API_KEY")
G_API_KEY = os.getenv("G_API_KEY")

client = genai.Client(api_key = G_API_KEY)
aai.settings.api_key = ASS_API_KEY

# Setting & Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

#Initialize Fast API
app = FastAPI(
    title = "AI Voice Agents",
    description = "30 Days of Voice Agents",
    version="1.0.0"
)

# All directories & paths
current_dir = Path(__file__).parent
static_path = current_dir / "static"
templates_path = current_dir / "templates"
uploads_path = current_dir/"uploads"

#Mounting Static files
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

#Pydantic models for request & response
class TTSRequest(BaseModel):
    text : str
    voiceid : str = "en-US-daniel"
    style: str = "Inspirational"
class TTSResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
class AudioUploadResponse(BaseModel):
    success: bool
    filename : Optional[str] = None
    content_type : Optional[str] = None
    size : Optional[int] = None
    message : Optional[str] = None
    error : Optional[str] = None
class TranscribeResponse(BaseModel):
    success : bool
    text : str
    message : Optional[str] = None
    error : Optional[str] = None
class EchoResponse(BaseModel):
    success: bool
    transcribed_text: Optional[str] = None
    audio_url: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
class LLMrequest(BaseModel):
    text : str = "Explain how AI works in a few words"
    model : str = "gemini-2.5-flash"
#Serving the HTML file
@app.get("/",response_class=HTMLResponse)
async def homepage():
    try:
        with open(templates_path/"index.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)   
    except FileNotFoundError:
        logger.error(f"index.html not found at {templates_path/"index.html"}")
        return HTMLResponse(content="<h1> index.html NOT FOUND</h1>")

# From Day1
@app.get("/api/data")
async def get_data():
    """Endpoint from Day 1 - for backward compatibility"""
    return {
        "message": "Hello from FastAPI backend!",
        "status": "success",
        "day": "Day 2 - TTS Integration",
        "framework": "FastAPI"
    }

# Convert text to speech in Murf Voice    
@app.post("/api/tts", response_model=TTSResponse)
async def tts(request: TTSRequest):

    # Checking API Key:
    if not MURF_API_KEY:
        logger.info("API_KEY not found in env file")
        raise HTTPException(
            status_code=500,
            detail="TTS service not configured. Please check API Key."
        )
    # Checking text input
    if not request.text.strip():
        raise HTTPException(
            status_code = 400,
            detail = "Text cannot be empty"
        )
    
    if len(request.text) > 5000:
        raise HTTPException(
            status_code = 400,
            detail = "Max 5000 characters allowed!"
        )
    
    #Making Request to Murf
    try:
        logger.info("Processing TTS Request")

        #Prepare two dictionary for request for Murf API
        header = {
            "api-key" : MURF_API_KEY,
            "Content-type" : "application/json"
        }
        payload = {
            "text" : request.text,
            "voiceId" : request.voiceid,
            "style" : request.style,
            "format" : "MP3",
            "sampleRate" : "8000.0"
        }

        #Calling Murf API:-
        logger.info(f"Calling MURF API at URL: {MURF_API_URL}")
        response = requests.post(
            MURF_API_URL,
            json = payload,
            headers = header,
            timeout = 45  #45 second timeout
        )

        #Handling API Response from Murf:

        if response.status_code == 200:
            response_data = response.json()
            audio = response_data.get("audioFile")
            
            if audio:
                logger.info("TTS generation successful")
                return TTSResponse(
                    success = True,
                    audio_url = audio,
                    message = "Text-to-speech Conversion successful!"
                )
            else:
                logger.error("No audio URL in Response")
                raise HTTPException(
                    status_code = 500,
                    detail = "TTS service returned invalid Response"
                )
        else:
            logger.error(f"Murf API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"TTS service error: {response.status_code}"
            )
    except requests.exceptions.Timeout:
        logger.error("Request to Murf API timed out")
        raise HTTPException(
        status_code=500,
        detail="TTS service request timed out"
    )
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Murf API")
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable. Please try again later."
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        ) 
    
# Convert Human voice to Murf Voice and also shows the Transcribed text
@app.post("/tts/echo", response_model=EchoResponse)
async def echo_tts(audio_file: UploadFile = File(...), voiceid: str = "en-US-daniel", style: str = "Inspirational"):
    """
    Echo Bot endpoint that transcribes audio and converts it back to speech using Murf TTS
    """
    # Check API keys
    if not ASS_API_KEY:
        logger.error("Assembly AI API key not found!")
        raise HTTPException(
            status_code=500,
            detail="Transcription service not configured"
        )
    
    if not MURF_API_KEY:
        logger.error("Murf API key not found!")
        raise HTTPException(
            status_code=500,
            detail="TTS service not configured"
        )

    if not audio_file:
        logger.error("No audio file received!")
        raise HTTPException(
            status_code=400,
            detail="No audio file provided"
        )

    # Save the uploaded audio file
    filename = f"echo_{int(time.time())}_{audio_file.filename}"
    filepath = uploads_path / filename

    try:
        # Save audio file
        logger.info("Saving audio file for echo processing...")
        with open(filepath, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Step 1: Transcribe the audio
        logger.info("Starting transcription...")
        transcript = aai.Transcriber().transcribe(str(filepath))

        if transcript.status == "error":
            logger.error(f"Transcription failed: {transcript.error}")
            raise HTTPException(
                status_code=400,
                detail=f"Transcription failed: {transcript.error}"
            )
        
        transcribed_text = transcript.text or ""

        if not transcribed_text or transcribed_text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the audio"
            )
        
        text_preview = transcribed_text[:50] if len(transcribed_text) > 50 else transcribed_text
        logger.info(f"Transcription successful: {text_preview}...")

        # Step 2: Convert transcribed text to speech using Murf
        logger.info("Converting transcribed text to speech...")
        
        headers = {
            "api-key": MURF_API_KEY,
            "Content-type": "application/json"
        }
        
        payload = {
            "text": transcribed_text,
            "voiceId": voiceid,
            "style": style,
            "format": "MP3",
            "sampleRate": "8000.0"
        }

        # Call Murf API
        response = requests.post(
            MURF_API_URL,
            json=payload,
            headers=headers,
            timeout=45
        )

        if response.status_code == 200:
            response_data = response.json()
            audio_url = response_data.get("audioFile")
            
            if audio_url:
                logger.info("Echo TTS generation successful")
                
                # Clean up the temporary file
                try:
                    filepath.unlink()
                except:
                    pass  # Ignore cleanup errors
                
                return EchoResponse(
                    success=True,
                    transcribed_text=transcribed_text,
                    audio_url=audio_url,
                    message="Echo TTS conversion successful!"
                )
            else:
                logger.error("No audio URL in Murf response")
                raise HTTPException(
                    status_code=500,
                    detail="TTS service returned invalid response"
                )
        else:
            logger.error(f"Murf API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"TTS service error: {response.status_code}"
            )

    except requests.exceptions.Timeout:
        logger.error("Request to Murf API timed out")
        raise HTTPException(
            status_code=500,
            detail="TTS service request timed out"
        )
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Murf API")
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error in echo endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Gemini API end point:-
@app.post("/llm/query")
async def llmquery(data : LLMrequest):
    if not G_API_KEY:
        logger.error("NO LLM KEY FOUND!")
        raise HTTPException(
            status_code=400,
            detail="NO API KEY IS NOT FOUND"
        )
    if not data.text:
        raise HTTPException(
            status_code=500,
            detail="Please Enter text"
        )
    
    logger.info(f"Calling Gemini API with model: {data.model}")
    logger.info(f"Query text: {data.text[:100]}...")
    
    try:
        g_response = client.models.generate_content(
            model = data.model,
            contents = data.text
            # Check Gemini docs to disable thinking
        )
        if g_response.candidates:
            logger.info("Success: The API returned a valid response with generated content.")
        else:
            logger.info("Failure: The API did not return any valid responses.")
            return
        if not g_response.text:
            logger.info("No response received from LLM")
            raise HTTPException(
                status_code=401,
                detail="No response received!"
            )
        else:
            return(g_response.text)
    except Exception as e:
        logger.info(f"LLM query error : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail = f"Internal server error: {str(e)}"
        )
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📍 Visit: http://127.0.0.1:8000")
    print("📖 API Docs: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)