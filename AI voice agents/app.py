# Importing packages
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import requests
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import Optional, List, Dict
import assemblyai as aai
import time
from google import genai
from google.genai import types
import uuid

# Loading all the environment files from .env
load_dotenv()

#Get API credentials from env file
MURF_API_KEY = os.getenv("MURF_API_KEY")
MURF_API_URL = os.getenv("MURF_API_URL","https://api.murf.ai/v1/speech/generate")
ASS_API_KEY = os.getenv("ASSEMBLY_API_KEY")
G_API_KEY = os.getenv("G_API_KEY")

if G_API_KEY:
    client = genai.Client(api_key = G_API_KEY)
if ASS_API_KEY:
    aai.settings.api_key = ASS_API_KEY

# Setting & Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

#Initialize Fast API
app = FastAPI(
    title = "AI Voice Agents",
    description = "AI Voice Agents with Chat History",
    version="2.0.0"
)

# All directories & paths
current_dir = Path(__file__).parent
static_path = current_dir / "static"
templates_path = current_dir / "templates"
uploads_path = current_dir/"uploads"

# Create uploads directory if it doesn't exist
uploads_path.mkdir(exist_ok=True)

#Mounting Static files
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# In-memory chat history storage (for prototype)
# Structure: {session_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

#Pydantic models for request & response
class LLMResponse(BaseModel):
    success: bool
    transcribed_text: Optional[str] = None
    llm_response: Optional[str] = None
    audio_url: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class ChatAgentResponse(BaseModel):
    success: bool
    session_id: str
    transcribed_text: Optional[str] = None
    llm_response: Optional[str] = None
    audio_url: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    message: Optional[str] = None
    error: Optional[str] = None

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
        "framework": "FastAPI"
    }

# Updated Gemini API endpoint for full LLM pipeline
@app.post("/llm/query", response_model=LLMResponse)
async def llm_query(audio_file: UploadFile = File(...), model: str = "gemini-2.0-flash-exp"):
    # Check all required API keys
    if not G_API_KEY:
        logger.error("NO LLM KEY FOUND!")
        raise HTTPException(
            status_code=500,
            detail="LLM API KEY NOT FOUND"
        )
    
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
        raise HTTPException(
            status_code=400,
            detail="Please provide audio file"
        )
    
    # Save the uploaded audio file
    filename = f"llm_{int(time.time())}_{audio_file.filename}"
    filepath = uploads_path / filename

    try:
        # Save audio file
        logger.info("Saving audio file for LLM processing...")
        with open(filepath, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Step 1: Transcribe the audio
        logger.info("Step 1: Transcribing audio...")
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
        
        # Step 2: Get LLM response
        logger.info(f"Step 2: Calling Gemini API with model: {model}")
        logger.info(f"Query text: {transcribed_text[:100]}...")
        
        g_response = client.models.generate_content(
            model=model,
            contents=transcribed_text
        )
        
        if not g_response.candidates:
            logger.error("Failure: The API did not return any valid responses.")
            raise HTTPException(
                status_code=500,
                detail="No response received from LLM"
            )
        
        if not g_response.text:
            logger.error("No response text received from LLM")
            raise HTTPException(
                status_code=500,
                detail="No response text received from LLM"
            )
        
        response_preview = g_response.text[:100] if len(g_response.text) > 100 else g_response.text
        logger.info(f"LLM response successful: {response_preview}...")
        
        # Step 3: Convert LLM response to speech using Murf
        logger.info("Step 3: Converting LLM response to speech...")
        
        headers = {
            "api-key": MURF_API_KEY,
            "Content-type": "application/json"
        }
        
        payload = {
            "text": g_response.text,
            "voiceId": "en-US-daniel",  # Fixed voice
            "style": "Conversational",
            "format": "MP3",
            "sampleRate": "8000.0"
        }

        # Call Murf API
        murf_response = requests.post(
            MURF_API_URL,
            json=payload,
            headers=headers,
            timeout=60  # Longer timeout for potentially longer responses
        )

        if murf_response.status_code == 200:
            murf_data = murf_response.json()
            audio_url = murf_data.get("audioFile")
            
            if audio_url:
                logger.info("Complete LLM pipeline successful!")
                
                # Clean up the temporary file
                try:
                    filepath.unlink()
                except:
                    pass  # Ignore cleanup errors
                
                return LLMResponse(
                    success=True,
                    transcribed_text=transcribed_text,
                    llm_response=g_response.text,
                    audio_url=audio_url,
                    message="LLM pipeline completed successfully!"
                )
            else:
                logger.error("No audio URL in Murf response")
                raise HTTPException(
                    status_code=500,
                    detail="TTS service returned invalid response"
                )
        else:
            logger.error(f"Murf API error: {murf_response.status_code} - {murf_response.text}")
            raise HTTPException(
                status_code=murf_response.status_code,
                detail=f"TTS service error: {murf_response.status_code}"
            )

    except requests.exceptions.Timeout:
        logger.error("Request timed out during LLM pipeline")
        raise HTTPException(
            status_code=500,
            detail="Request timed out. Please try again."
        )
    except requests.exceptions.ConnectionError:
        logger.error("Connection error during LLM pipeline")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. Please try again later."
        )
    except Exception as e:
        logger.error(f"LLM pipeline error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# NEW: Chat Agent endpoint with session management and chat history
@app.post("/agent/chat/{session_id}", response_model=ChatAgentResponse)
async def chat_agent(
    session_id: str,
    audio_file: UploadFile = File(...),
    model: str = "gemini-2.0-flash-exp",
    disable_thinking: bool = False
):
    """
    Chat Agent with persistent session history
    """
    # Check all required API keys
    if not G_API_KEY:
        logger.error("NO LLM KEY FOUND!")
        raise HTTPException(
            status_code=500,
            detail="LLM API KEY NOT FOUND"
        )
    
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
        raise HTTPException(
            status_code=400,
            detail="Please provide audio file"
        )
    
    # Initialize session if it doesn't exist
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        logger.info(f"Created new chat session: {session_id}")
    
    # Save the uploaded audio file
    filename = f"chat_{session_id}_{int(time.time())}_{audio_file.filename}"
    filepath = uploads_path / filename

    try:
        # Save audio file
        logger.info(f"Processing chat for session: {session_id}")
        with open(filepath, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Step 1: Transcribe the audio
        logger.info("Transcribing user audio...")
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
        
        logger.info(f"User said: {transcribed_text[:100]}...")
        
        # Step 2: Add user message to chat history
        chat_sessions[session_id].append({
            "role": "user",
            "content": transcribed_text
        })
        
        # Step 3: Prepare context for LLM (last 10 messages to keep context reasonable)
        recent_history = chat_sessions[session_id][-10:]  # Keep last 10 messages
        
        # Build context string for Gemini
        context_messages = []
        for msg in recent_history[:-1]:  # All except the last one (current user message)
            if msg["role"] == "user":
                context_messages.append(f"User: {msg['content']}")
            else:
                context_messages.append(f"Assistant: {msg['content']}")
        
        # Create full prompt with context
        context_str = "\n".join(context_messages) if context_messages else ""
        full_prompt = f"{context_str}\nUser: {transcribed_text}" if context_str else transcribed_text
        
        logger.info(f"Sending to LLM with context length: {len(context_str)} characters")
        
        # Step 4: Get LLM response
        logger.info(f"Getting LLM response with model: {model}")
        
        # Configure generation options based on disable_thinking flag
        generation_config = {}
        if disable_thinking:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024
            }
        
        g_response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ) if disable_thinking else None
        )
        
        if not g_response.candidates or not g_response.text:
            logger.error("No valid response from LLM")
            raise HTTPException(
                status_code=500,
                detail="No response received from LLM"
            )
        
        llm_response_text = g_response.text
        logger.info(f"LLM responded: {llm_response_text[:100]}...")
        
        # Step 5: Add LLM response to chat history
        chat_sessions[session_id].append({
            "role": "assistant", 
            "content": llm_response_text
        })
        
        # Step 6: Convert LLM response to speech
        logger.info("Converting LLM response to speech...")
        
        headers = {
            "api-key": MURF_API_KEY,
            "Content-type": "application/json"
        }
        
        payload = {
            "text": llm_response_text,
            "voiceId": "en-US-daniel",  # Fixed voice
            "style": "Conversational",
            "format": "MP3",
            "sampleRate": "8000.0"
        }

        murf_response = requests.post(
            MURF_API_URL,
            json=payload,
            headers=headers,
            timeout=60
        )

        if murf_response.status_code == 200:
            murf_data = murf_response.json()
            audio_url = murf_data.get("audioFile")
            
            if audio_url:
                logger.info(f"Chat agent pipeline completed for session: {session_id}")
                
                # Clean up the temporary file
                try:
                    filepath.unlink()
                except:
                    pass
                
                return ChatAgentResponse(
                    success=True,
                    session_id=session_id,
                    transcribed_text=transcribed_text,
                    llm_response=llm_response_text,
                    audio_url=audio_url,
                    chat_history=chat_sessions[session_id],
                    message="Chat completed successfully!"
                )
            else:
                logger.error("No audio URL in Murf response")
                raise HTTPException(
                    status_code=500,
                    detail="TTS service returned invalid response"
                )
        else:
            logger.error(f"Murf API error: {murf_response.status_code}")
            raise HTTPException(
                status_code=murf_response.status_code,
                detail="TTS service error"
            )

    except requests.exceptions.Timeout:
        logger.error("Request timed out during chat agent pipeline")
        raise HTTPException(
            status_code=500,
            detail="Request timed out. Please try again."
        )
    except requests.exceptions.ConnectionError:
        logger.error("Connection error during chat agent pipeline")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. Please try again later."
        )
    except Exception as e:
        logger.error(f"Chat agent pipeline error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Get chat history for a session
@app.get("/agent/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a specific session"""
    if session_id not in chat_sessions:
        return {
            "success": True,
            "session_id": session_id,
            "chat_history": [],
            "message": "New session created"
        }
    
    return {
        "success": True,
        "session_id": session_id,
        "chat_history": chat_sessions[session_id],
        "message": "Chat history retrieved successfully"
    }

# Clear chat history for a session
@app.delete("/agent/chat/{session_id}/history")
async def clear_chat_history(session_id: str):
    """Clear chat history for a specific session"""
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
        logger.info(f"Cleared chat history for session: {session_id}")
    
    return {
        "success": True,
        "session_id": session_id,
        "message": "Chat history cleared successfully"
    }

# Get all active sessions
@app.get("/agent/sessions")
async def get_active_sessions():
    """Get all active chat sessions"""
    active_sessions = [
        {
            "session_id": session_id,
            "message_count": len(messages),
            "last_updated": "N/A"  # Could add timestamps later
        }
        for session_id, messages in chat_sessions.items()
    ]
    
    return {
        "success": True,
        "active_sessions": active_sessions,
        "total_sessions": len(active_sessions)
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced FastAPI server with Chat History...")
    print("📍 Visit: http://127.0.0.1:8000")
    print("📖 API Docs: http://127.0.0.1:8000/docs")

    # Print API key status for debugging
    print(f"🔑 MURF_API_KEY configured: {'Yes' if MURF_API_KEY else 'No'}")
    print(f"🔑 AAI_API_KEY configured: {'Yes' if ASS_API_KEY else 'No'}")
    print(f"🔑 G_API_KEY configured: {'Yes' if G_API_KEY else 'No'}")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, workers=1)