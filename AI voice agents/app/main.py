import requests
from urllib.parse import quote
import json
import asyncio
import websockets
import base64
import assemblyai as aai
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from app import config
from pathlib import Path
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions,
    StreamingParameters, StreamingSessionParameters,
    StreamingEvents, BeginEvent, TurnEvent,
    TerminationEvent, StreamingError
)
import re

# Initialize services
aai.settings.api_key = config.ASSEMBLY_API_KEY

app = FastAPI()
STATIC_DIR = Path(__file__).parent.parent / "static"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "index.html"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class StreamManager:
    def __init__(self):
        self.api_keys={}
        self.client = None
        self.websocket = None
        self.current_turn = ""
        self.loop = None
        self.chatHistory = []
    
    def updateApiKeys(self, keys):
        self.api_keys = keys
        if keys.get('assemblyai'):
            aai.settings.api_key = keys['assemblyai']

        
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

    async def _search_web(self, query):
        """Search the web and return results"""
        try:
            # If no search API key is configured, use a mock response
            if not hasattr(config, 'SEARCH_API_KEY') or not config.SEARCH_API_KEY:
                return f"Mock search results for '{query}':\n• Example result 1\n• Example result 2\n• Example result 3"
                
            # Example using SerpAPI (you can use any search API)
            url = f"https://serpapi.com/search.json?q={quote(query)}&api_key={config.SEARCH_API_KEY}"
            response = requests.get(url)
            data = response.json()
        
            # Extract top 3-5 results
            results = []
            for result in data.get('organic_results', [])[:3]:
                results.append(f"• {result.get('title', '')}: {result.get('snippet', '')}")
        
            return "\n".join(results) if results else "No search results found."
        except Exception as e:
            print(f"Search error: {e}")
            return f"Search error: {str(e)}"
        
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
            # Improved search detection with more patterns
            search_patterns = [
                r"search (?:for|about) (.+)",
                r"look up (.+)",
                r"find (?:information|details) (?:on|about) (.+)",
                r"what is (.+)",
                r"who is (.+)",
                r"how to (.+)",
                r"tell me about (.+)",
                r"explain (.+)",
                r"web search (.+)",
                r"internet search (.+)"
            ]
            
            search_query = None
            context = ""
            
            # Check if any search pattern matches
            for pattern in search_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    search_query = match.group(1).strip()
                    break
            
            # If we found a search query, perform the search
            if search_query:
                await self._send_to_websocket({"type": "searching"})
                search_results = await self._search_web(search_query)
                context = f"Search results for '{search_query}':\n{search_results}"
            
            # Add user chat to history
            self.chatHistory.append({"role":"user","content":text})

            conversation_context = ""
            for msg in self.chatHistory[-10:]:
                role = "Human" if msg["role"] == "user" else "System"
                conversation_context += f"{role}:{msg['content']}\n"

            prompt = f"""You are a friendly and helpful AI assistant.
                You are playing a role named System in Solo Levelling
                Act as system and the person chatting with you is the Greatest Sung Jinwoo also known as Hunter Sung 
                Here's our conversation history : {conversation_context}
                
                {f"Additional context from web search: {context}" if context else ""}
                
                You've to respond like system in Solo leveling but if sung jinwoo asks for extra info
                like searching for the web for any info Then you just have to sound like SYSTEM and give
                any info Hunter sung demands provide it as all llm provide and search on web.
                If you were provided search results, incorporate them naturally into your response.
                However you can answer the question that are not related to Solo leveling"""

            llm_text = call_gemini_api(prompt, config.GEMINI_API_KEY)

            # Add assistant response to history
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
                    if data.get("apiKeys"):
                        stream_manager.updateApiKeys(data["apiKeys"])
                    print("▶️ Starting transcription session")
                    await websocket.send_text(json.dumps({"type": "ready"}))
            
            elif "bytes" in message:
                stream_manager.send_audio(message["bytes"])
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 WebSocket connection closing")
        stream_manager.stop()

def call_gemini_api(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    response.raise_for_status()
    result = response.json()
    # Extract the response text
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "Sorry, I couldn't get a response from Gemini."

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting server at http://{config.HOST}:{config.PORT}")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)