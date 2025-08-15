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
├── templates/
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

### **Day 12: Enhanced UI and also removed echo bot & AI voice generator
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

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)
- **Frontend:** HTML, CSS, JavaScript
- **APIs Used:**  
  - Murf API (Text-to-Speech)  
  - AssemblyAI (Speech-to-Text)  
  - Google Gemini API (LLM)
- **Other Tools:** `.env` for secrets

---
