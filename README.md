# 30 Days of AI Voice Agents Challenge by Murf AI 🚀

This repository documents my progress in the **#30DaysofVoiceAgents** challenge by **Murf AI**.  
Each day focuses on building a fully functional AI-powered voice application, step-by-step, using **FastAPI**, **JavaScript**, **HTML**, **Murf API**, **AssemblyAI**, and **Google Gemini API**.


## Project Structure
```
AI voice agents/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py
│   │   ├── stt_service.py
│   │   └── tts_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── session_manager.py
│   │   └── config.py
│   └── main.py
├── static/
│   ├── script.js
│   └── styles.css
├── templates/
│   └── index.html
├── uploads/
├── .env
├── .gitignore
├── app.log
├── README.md
├── requirements.txt
```

---

## 📅 Daily Progress

### **Day 1: Project Setup**
- Initialized a **Python FastAPI** backend.
- Created a basic `index.html` and `script.js` for the frontend.
- Served the HTML page from the Python server.

**Skills Learned:**  
✅ FastAPI setup  
✅ Static file serving in FastAPI  

---

### **Day 2: Your First REST TTS Call**
- Created `/generate-audio` endpoint.
- Integrated **Murf REST TTS API** to convert input text to speech.
- Returned an **audio URL** in the API response.
- Secured API keys using `.env`.

**Skills Learned:**  
✅ REST API integration  
✅ Environment variable security  

---

### **Day 3: Playing Back TTS Audio**
- Added a **text input** and **submit button** in the frontend.
- Sent the text to `/generate-audio` endpoint.
- Played the returned audio using `<audio>` tag in HTML.

**Skills Learned:**  
✅ Fetch API in JavaScript  
✅ HTML audio playback  

---

### **Day 4: Echo Bot v1**
- Added **"Echo Bot"** section in UI.
- Used the **MediaRecorder API** to record audio from the microphone.
- Created **Start** and **Stop** buttons to control recording.
- Played back recorded audio in `<audio>` element.

**Skills Learned:**  
✅ MediaRecorder API  
✅ Handling audio blobs in JS  

---

### **Day 5: Sending Audio to the Server**
- Modified Echo Bot to **upload recorded audio** to the server after stopping recording.
- Built `/upload-audio` endpoint in FastAPI.
- Saved uploaded audio in `/uploads` folder.
- Returned **file name, content type, and size** in response.

**Skills Learned:**  
✅ File uploads in FastAPI  
✅ Handling `UploadFile` in Python  

---

### **Day 6: Server-Side Transcription**
- Created `/transcribe/file` endpoint.
- Integrated **AssemblyAI** to transcribe uploaded audio.
- Displayed transcription in the UI.

**Skills Learned:**  
✅ AssemblyAI transcription API  
✅ Working with binary audio data in Python  

---

### **Day 7: Echo Bot v2**
- Created `/tts/echo` endpoint.
- Transcribed uploaded audio using AssemblyAI.
- Sent transcription to Murf API to generate speech in a Murf voice.
- Played Murf-generated audio in the Echo Bot UI.

**Skills Learned:**  
✅ Combining APIs for multi-step workflows  
✅ Real-time audio transformation  

---

### **Day 8: Integrating a Large Language Model (LLM)**
- Created `/llm/query` endpoint.
- Integrated **Google Gemini API** to generate responses to user input.
- Returned LLM output as API response.

**Skills Learned:**  
✅ LLM API integration  
✅ Building conversational endpoints  

---

### **Day 9: The Full Non-Streaming Pipeline**
- Updated `/llm/query` endpoint.
- Accept audio as input.
- Audio is transcribed through **AssemblyAI**
- Transciption sent to **Google Gemini API**
- **Gemini API** text response is converted into Murf Voice using **Murf API**

**Skils Learned:**
✅ Combining LLM API with other for multi-step workflows  
✅ Real-time audio response from LLM

---

### **Day 10: Chat History for Conversational Bot**
- Created a POST `/agent/chat/{session_id}` endpoint.
- Used an in-memory dictionary datastore to store chat history per session_id.
- Chat history for that session_id is fetched and combined with the new message and sent to LLM
- LLM response is stored back into chat history and audio response is given to user.
-Automatically starts recording after an LLM audio response

**Skills Learned:**
✅ Session-based conversation memory
✅ Combining STT → LLM → TTS in one flow
✅ Auto-triggering recording in the frontend

---

### **Day 11: Building Resilient AI Voice Applications: Mastering Error Handling 🚨**
🔥 Scenarios I tackled:
- API & Service Failures
- Audio Processing Issues
- Network & Connectivity Problems
- Session Stability

**Skills Learned:**
✅ More efficient error handling

---

### **Day 12: Enhanced UI and also removed echo bot & AI voice generator**
- ➕ “New Session” button 
- Changed the old looking UI to modern LLM UI's like Gemini, ChatGPT,etc 
- Single toggle start/stop recording button
- ❌ Stop Conversation button at bottom right — stops auto-mic loop after agent responses
- Transcribed conversation text displayed above buttons, along with LLM status updates
- Removed initial TTS & Echo Bot sections for a cleaner layout

**Skills Learned:**
✅ Building more convenient UI
✅ Making UI like Modern LLMs

---

### **Day 13: Create a Readme file if not created yet**
- Already Done!

---

### **Day 14: Code Refactoring & Project Cleanup 🛠️**
- Separated **schemas** (Pydantic models) for request/response objects into `/models` for cleaner endpoint definitions.
- Moved **STT**, **TTS**, and **LLM** integrations into dedicated files under `/services`.
- Added `session_manager.py` in `/utils` for managing chat sessions and history.
- Added `config.py` in `/main` for centralized configuration and environment variable management.
- Created `__init__.py` in every folder for proper Python package structure.
- Removed unused imports, variables, and functions.
- Improved code readability with clear function names and docstrings.
- Updated README with latest project structure.

**Skills Learned:**
✅ Code refactoring best practices  
✅ Organizing a FastAPI project for maintainability  
✅ Using Python’s `logging` module effectively  
✅ Writing cleaner, more modular code

---
### **Streaming Branch** 
---

### **Day 15: WebSocket Connection**
- Created a **`/ws` WebSocket endpoint** in FastAPI.  
- Established real-time connection between **client ↔ server**.  
- Implemented echo functionality: server sends back the same message received from client (tested via Postman).  
- Worked on a **separate `streaming` branch** to keep non-streaming code safe.  

**Skills Learned:**  
✅ FastAPI WebSocket setup  
✅ Real-time message echo testing  

---

### **Day 16: Streaming Audio**
- Extended recording logic on client to **stream audio chunks via WebSockets** at intervals.  
- Server received binary audio data through WebSocket and **saved it to a file**.  
- Verified audio reception and storage (no transcription/LLM yet).  

**Skills Learned:**  
✅ Streaming binary data over WebSockets  
✅ Audio file handling on the server  

---

### **Day 17: WebSockets + AssemblyAI**
- Integrated **AssemblyAI Python SDK** with streaming WebSocket audio.  
- Converted audio into required **16kHz, 16-bit, mono PCM format**.  
- Transcriptions were generated in real-time and printed to console (or UI).  

**Skills Learned:**  
✅ Live transcription with AssemblyAI  
✅ Audio preprocessing for STT  

---

### **Day 18: Turn Detection**
- Used AssemblyAI’s **streaming API with turn detection**.  
- Detected when the user stopped talking.  
- Server notified client of **end-of-turn events** via WebSocket.  
- Displayed **finalized transcription** on the UI after each turn.  

**Skills Learned:**  
✅ Turn detection in speech streams  
✅ Real-time server → client updates via WebSocket  

---

### **Day 19: Streaming LLM Responses**
- After receiving a finalized transcript → sent it to **Gemini LLM API**.  
- Used `streamGenerateContent` to **stream LLM response chunks**.  
- Accumulated responses and printed them to console (no UI yet).  

**Skills Learned:**  
✅ Streaming responses from LLM  
✅ Handling incremental response chunks 

---

### **Day 20: Streaming LLM → Murf Voice via WebSockets**
- Integrated **Gemini LLM streaming responses** with **Murf TTS** over WebSockets.  
- Workflow:  
  1. Sent **streaming LLM responses** (from Gemini) to Murf WebSocket API.  
  2. Murf returned **audio in base64 format**.  
  3. Printed encoded audio directly in server console for verification.  
  4. Used a static **`context_id`** in Murf WebSocket requests to avoid context overflow.  

**Skills Learned:**  
✅ Streaming text → speech in real-time  
✅ Handling base64 audio over WebSockets  
✅ Managing Murf WebSocket sessions with context_id  

---

### **Day 21: Streaming Audio Data to the Client**
- Extended WebSocket logic to **send streaming audio data from server → client** in real time.  
- On the client side:  
  - **Accumulated base64 chunks** into an array.  
  - Printed **acknowledgements** on console confirming successful reception of audio data.  

**Skills Learned:**  
✅ Server → client streaming over WebSockets  
✅ Handling real-time base64 audio data  
✅ Ensuring reliable audio delivery with acknowledgements  

---

### **Day 22 – Seamless Real-Time Audio Playback 🔊⚡**
- Implemented **real-time audio playback** in the browser.
- Received and buffered base64 audio chunks from the server.
- Stitched chunks together for **smooth, continuous playback**.
- Eliminated pauses → AI voice now flows naturally.

**Skills Learned:**  
✅ Handling **audio streams** efficiently in the browser.  
✅ Buffering + decoding **base64 audio chunks** in real time.  
✅ Achieving **low-latency playback** for seamless user experience.

---

### **Day 23 – Complete Conversational Agent 🤖🎙️**
- Integrated all components into a single end-to-end pipeline:
  - Capture + stream **user audio** (client → server).
  - **Real-time transcription** via AssemblyAI.
  - Pass transcript to **Gemini LLM** for reasoning.
  - Maintain **chat history** for context.
  - Generate natural voice using **Murf TTS**.
  - Stream **audio back to the client** for playback.
- Achieved a fully functional, streaming conversational agent.

**Skills Learned:**  
✅ Designing a **modular, end-to-end system** with multiple APIs.  
✅ Managing **real-time data flow** between client and server.  
✅ Synchronizing **voice, transcription, and LLM reasoning** for fluid conversations.

---

### **Day 24 – Adding a Persona 🧙⚔️**
- Introduced **System persona** for the LLM.
- Persona inspired by *Solo Leveling* ⚡ → gives the agent a unique tone + style.
- Conversations feel more **immersive and engaging**.
- Set foundation for **multi-persona conversational agents**.

**Skills Learned:**  
✅ Prompt engineering with **system personas** for LLMs.  
✅ Controlling **tone, style, and character** in AI responses.  
✅ Understanding the role of **personas** in making AI interactions engaging + human-like.

---
### **Day 25 – Agent Special Skill 1: Web Search 🌐🤖**  
- Integrated **web search capability** into the agent.  
- User queries can now be routed through the LLM → external web search.  
- Responses are generated with **fresh, real-time knowledge**, not just static training data.  

**Skills Learned:**  
✅ Connecting LLM with external web search  
✅ Enabling **real-time information retrieval**  
✅ Expanding agent capabilities beyond training data  

---  

### **Day 26 – Agent Special Skill 2: Weather Updates ☀️🌧️**  
- Leveraged the **Web Search integration** to provide real-time weather updates.  
- Users can ask about **current conditions, temperature, or forecasts** in natural language.  
- Demonstrated how the same web search skill can power **practical, everyday use cases** like weather.  

**Skills Learned:**  
✅ Extending existing skills (Web Search) to handle new use cases  
✅ Parsing + presenting **structured real-world data** (weather info)  
✅ Building a more **helpful, context-aware assistant**  

---

### **Day 27 – Revamp UI + Code Cleanup 🎨⚡**  
- Added a **config section in the UI** to let users enter their own API keys.  
- Revamped the UI for a **cleaner, modern look**.  
- Performed **code cleanup + refactoring** for better maintainability.  
- Left room for adding any final features.  

**Skills Learned:**  
✅ Improving **UI/UX** for better usability  
✅ Configuring apps to accept **user-provided API keys**  
✅ Refactoring + maintaining cleaner codebases  

---  

### **Day 28 – Deploying the Agent to the Cloud ☁️🚀**  
- Deployed the voice agent to a **cloud hosting service** (free tier).  
- Configured environment + dependencies for production.  
- Made the agent accessible via a **public endpoint**.  
- Ensured the deployment is **lightweight, scalable, and reliable**.  

**Skills Learned:**  
✅ Cloud deployment basics  
✅ Making apps **publicly accessible**  
✅ Managing hosting, dependencies, and free-tier resources safely  

---
### **Day 29 - Update Readme.md file**
---

## 🛠️ Tech Stack  

### **Backend**  
- **FastAPI (Python)** – Core backend framework  
- **WebSockets (FastAPI + WebSocket API)** – Real-time audio & text streaming  
- **uvicorn** – ASGI server for running FastAPI apps  

### **Frontend**  
- **HTML, CSS, JavaScript**  
- **WebSocket API (Browser)** – Sending/receiving streaming data  
- **Modernized UI** – Revamped interface with recording controls, session handling, and API key input  

### **AI & APIs**  
- **Murf API** – Text-to-Speech (REST + WebSocket)  
- **AssemblyAI API** – Speech-to-Text (file upload + streaming + turn detection)  
- **Google Gemini API** – Large Language Model (text + streaming responses)  
- **Web Search Integration** – Fetches real-time knowledge from the internet, also used for **weather updates**  

### **Data Handling**  
- **In-memory datastore (Python dict)** – Session-based chat history  
- **Base64 encoding** – Audio transmission  
- **PCM Audio (16kHz, 16-bit, mono)** – Format required for STT streaming  
- **File uploads & storage** – Temporary audio storage for testing  

### **Other Tools & Deployment**  
- **dotenv** – API key management via `.env` (plus user-provided API key config in UI)  
- **Postman** – API testing  
- **Git + GitHub** – Version control  
- **Branching strategy** – Separate `streaming` branch for WebSocket features  
- **Cloud Hosting (Render/Free-tier providers)** – Public deployment of the voice agent  

---
