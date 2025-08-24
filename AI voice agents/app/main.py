import json
import asyncio
import websockets
import base64
import assemblyai as aai
from google import genai
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import config
from pathlib import Path
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions,
    StreamingParameters, StreamingSessionParameters,
    StreamingEvents, BeginEvent, TurnEvent,
    TerminationEvent, StreamingError
)

# Initialize services
aai.settings.api_key = config.ASSEMBLY_API_KEY
gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

app = FastAPI()
STATIC_DIR = Path(__file__).parent.parent / "static"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "index.html"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class StreamManager:
    def __init__(self):
        self.client = None
        self.websocket = None
        self.current_turn = ""
        self.loop = None
        self.chatHistory = []
        
    async def start_transcription(self, websocket):
        self.websocket = websocket
        self.loop = asyncio.get_running_loop()
        
        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=config.ASSEMBLY_API_KEY,
                api_host="streaming.assemblyai.com"
            )
        )
        
        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_termination)
        self.client.on(StreamingEvents.Error, self.on_error)
        
        self.client.connect(
            StreamingParameters(sample_rate=16000, format_turns=False)
        )
        
    def on_begin(self, client, event: BeginEvent):
        print(f"🎤 Session started: {event.id}")
        
    def on_turn(self, client, event: TurnEvent):
        print(f"📝 Transcript: {event.transcript} (end_of_turn={event.end_of_turn})")
        
        if event.end_of_turn and event.transcript.strip():
            self.current_turn = event.transcript
            print(f"🎯 Final transcript: {event.transcript}")
            if self.loop and not self.loop.is_closed():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._process_turn(), 
                        self.loop
                    )
                except RuntimeError as e:
                    print(f"Error scheduling coroutine: {e}")
            
    async def _process_turn(self):
        if self.current_turn:
            await self._send_to_websocket({"type": "turn_end", "text": self.current_turn})
            await self._get_llm_response(self.current_turn)
            self.current_turn = ""
    
    async def _get_llm_response(self, text):
        print(f"🤖 Sending to LLM: {text}")
        try:
            # Add user chat to history
            self.chatHistory.append({"role":"user","content":text})

            conversation_context = ""
            for msg in self.chatHistory[-10:]:
                role = "Human" if msg["role"] == "user" else "System"
                conversation_context += f"{role}:{msg['content']}\n"

            prompt = f"""You are a friendly and helpful AI assistant.
                        Your Name is System. 
                        And You are Created, Developed and trained by Kartik Garg
                        Here's our conversation hsitory : {conversation_context}
                        Respond naturally to the latest message."""

            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            llm_text = response.text

            #Add assistant response to history
            self.chatHistory.append({"role":"System","content":llm_text})

            print(f"🤖 LLM Response: {llm_text}")
            await self._send_to_websocket({"type": "llm_response", "text": llm_text})
            
            # Send to Murf for TTS
            await self._send_to_murf(llm_text)
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            await self._send_to_websocket({
                "type": "llm_response", 
                "text": f"Sorry, I encountered an error: {str(e)}"
            })
    
    async def _send_to_murf(self, text):
        """Send text to Murf WebSocket for TTS and get base64 audio"""
        try:
            murf_ws_url = "wss://api.murf.ai/v1/speech/stream-input"
            headers = {"api-key": f"{config.MURF_API_KEY}"}
        
            print(f"🔗 Connecting to Murf WebSocket...")
            async with websockets.connect(murf_ws_url, additional_headers=headers) as murf_ws:
            
                request = {
                    "context_id": f"turn_{int(asyncio.get_event_loop().time())}",
                    "text": text,
                    "voice_config": {
                        "voice_id" : "en-US-Daniel",
                        "style":"Inspirational"},
                    "format": "mp3",
                    "sample_rate": 24000
                }
            
                await murf_ws.send(json.dumps(request))
                print(f"🎵 Sent to Murf: {text[:50]}...")
                text_msg = {
                    "text" : text,
                    "end" :True
                }
                await murf_ws.send(json.dumps(text_msg))
                while True:
                    response = await murf_ws.recv()
                    audio_data = json.loads(response)
                    if "audio" in audio_data:
                        print("Murf Audio:",audio_data["audio"][:30],"...")
                        base64_audio = base64.b64decode(audio_data["audio"])
                        if len(base64_audio)>44:
                            base64_audio = f"{base64_audio[:30]} ..."
                            print(f"Base64 audio{base64_audio}")
                        await self._send_to_websocket({"type" : "audio_chunk", "data" : audio_data["audio"]})
                    if audio_data.get("final"):
                        break
        except Exception as e:
            print(f"❌ Murf WebSocket Error: {e}")
    
    def on_termination(self, client, event: TerminationEvent):
        print(f"🛑 Session terminated after {event.audio_duration_seconds} s")
        
    def on_error(self, client, error: StreamingError):
        print(f"❌ AssemblyAI Error: {error}")
        
    async def _send_to_websocket(self, data):
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps(data))
            except Exception as e:
                print(f"❌ WebSocket send error: {e}")
    
    def send_audio(self, audio_data):
        if self.client and len(audio_data) > 0:
            print(f"📡 Sending audio chunk: {len(audio_data)} bytes")
            self.client.stream(audio_data)
    
    def stop(self):
        if self.client:
            self.client.disconnect(terminate=True)
    
    def reset_chat_history(self):
        self.chatHistory = []
        print("🔄 Chat history reset")

stream_manager = StreamManager()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(str(TEMPLATES_DIR), encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket connection established")
    
    try:
        await stream_manager.start_transcription(websocket)
        
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "start":
                    print("▶️ Starting transcription session")
                    await websocket.send_text(json.dumps({"type": "ready"}))
            
            elif "bytes" in message:
                stream_manager.send_audio(message["bytes"])
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 WebSocket connection closing")
        stream_manager.stop()

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting server at http://{config.HOST}:{config.PORT}")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)