let ws = null;
let isRecording = false;
let audioBuffer = [];
let audioContext = null;
let playbackAudioContext = null;
let processor = null;
let source = null;
let stream = null;
let isPlayingAudio = false;
let playheadTime = 0;
let finaltext;

const SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 100;
const SAMPLES_PER_CHUNK = (SAMPLE_RATE * CHUNK_DURATION_MS) / 1000;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const messages = document.getElementById('messages');

let apiKeys = {};

// Config modal handlers
document.getElementById('configBtn').onclick = () => {
    document.getElementById('configModal').classList.remove('hidden');
};

document.getElementById('closeConfig').onclick = () => {
    document.getElementById('configModal').classList.add('hidden');
};

document.getElementById('saveConfig').onclick = () => {
    apiKeys = {
        assemblyai: document.getElementById('assemblyKey').value,
        gemini: document.getElementById('geminiKey').value,
        murf: document.getElementById('murfKey').value,
        search: document.getElementById('searchKey').value
    };
    document.getElementById('configModal').classList.add('hidden');
    addMessage('🔧 SYSTEM: API keys configured');
};

startBtn.onclick = startRecording;
stopBtn.onclick = stopRecording;

function getWebSocketURL() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${location.host}/ws/chat`;
}

function sendAudioChunk() {
    if (audioBuffer.length === 0 || !ws || ws.readyState !== WebSocket.OPEN) return;
    const totalSamples = audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
    if (totalSamples < SAMPLES_PER_CHUNK) return; // Wait for more data
    const chunkBuffer = new Int16Array(SAMPLES_PER_CHUNK);
    let samplesUsed = 0;
    let bufferIndex = 0;

    while (samplesUsed < SAMPLES_PER_CHUNK && bufferIndex < audioBuffer.length) {
        const currentChunk = audioBuffer[bufferIndex];
        const samplesNeeded = SAMPLES_PER_CHUNK - samplesUsed;
        const samplesToTake = Math.min(samplesNeeded, currentChunk.length);

        chunkBuffer.set(currentChunk.subarray(0, samplesToTake), samplesUsed);
        samplesUsed += samplesToTake;

        if (samplesToTake < currentChunk.length) {
            audioBuffer[bufferIndex] = currentChunk.subarray(samplesToTake);
        } else {
            bufferIndex++;
        }
    }

    audioBuffer = audioBuffer.slice(bufferIndex);
    console.log(`Sending audio chunk: ${chunkBuffer.length} samples (${CHUNK_DURATION_MS}ms)`);
    const maxSamples = (SAMPLE_RATE * 800) / 1000;
    if (chunkBuffer.length > maxSamples) {
        console.log("Chunk too large, skipping");
        return;
    }
    ws.send(chunkBuffer.buffer);
}

async function startRecording() {
    try {
        audioBuffer = [];
        if (!apiKeys.assemblyai || !apiKeys.gemini || !apiKeys.murf) {
            alert('Please configure API keys first (click ⚙️ CONFIG button)');
            document.getElementById('configModal').classList.remove('hidden');
            return;
        }
        ws = new WebSocket(getWebSocketURL()); // Use dynamic URL
        ws.onopen = () => {
            status.textContent = 'SYSTEM ONLINE';
            status.className = 'status connected';
            ws.send(JSON.stringify({type: 'start', apiKeys:apiKeys}));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Received:', data.type);
            handleMessage(data);
        };
        
        ws.onclose = () => {
            status.textContent = 'SYSTEM OFFLINE';
            status.className = 'status disconnected';
        };

        ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            status.textContent = 'CONNECTION ERROR';
            status.className = 'status disconnected';
        };

        stream = await navigator.mediaDevices.getUserMedia({
            audio: { 
                sampleRate: SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        audioContext = new (window.AudioContext)({
            sampleRate: SAMPLE_RATE
        });

        processor = audioContext.createScriptProcessor(4096, 1, 1);
        source = audioContext.createMediaStreamSource(stream);

        processor.onaudioprocess = (event) => {
            if (!ws || ws.readyState !== WebSocket.OPEN || isPlayingAudio || playheadTime > audioContext.currentTime) return;
            
            const inputData = event.inputBuffer.getChannelData(0);
            const int16Data = new Int16Array(inputData.length);
    
            for (let i = 0; i < inputData.length; i++) {
                const sample = Math.max(-1, Math.min(1, inputData[i]));
                int16Data[i] = sample * 0x7FFF;
            }
    
            audioBuffer.push(int16Data);
            sendAudioChunk();
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
        
        isRecording = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        startBtn.textContent = '🎙️ RECORDING...';
        
        console.log(`Started recording with ${CHUNK_DURATION_MS}ms chunks`);
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Microphone access denied or not available');
        resetUI();
    }
}

function stopRecording() {
    console.log('Stopping recording...');
    
    audioBuffer = [];
    isPlayingAudio = false;
    playheadTime = 0;
    
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (source) {
        source.disconnect();
        source = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        console.log('Audio context closed');        
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    resetUI();
}

function resetUI() {
    isRecording = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    startBtn.textContent = 'INITIATE RECORDING';
    status.textContent = 'SYSTEM OFFLINE';
    status.className = 'status disconnected';
}

function handleMessage(data) {
    if (data.type === 'audio_chunk' && data.data) {
        if (isPlayingAudio) return;
        playAudioChunk(data.data);
    } else if (data.type === 'turn_end') {
        addMessage(`👤 You: ${data.text}`);
    } else if (data.type === 'llm_response') {
        const isSearchResponse = data.text.includes('Search results') || data.text.includes('I found');
        addMessage(`🤖 SYSTEM: ${data.text}`, 'llm-response');
    } else if (data.type === 'searching') {
        addMessage(`🔍 SYSTEM: Searching the web...`, 'search-status');
    }
}

function addMessage(text, className = '') {
    finaltext = text.replace(/(\r\n|\r|\n)/g, "<br>");
    finaltext = finaltext.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>'); // Replace bold (**text**) 
    finaltext = finaltext.replace(/\*(.*?)\*/g, '<i>$1</i>'); // Replace italics (*text*)
    console.log("Final: ",finaltext);
    const div = document.createElement('div');
    div.className = `turn ${className}`;
    div.innerHTML = `<div>${finaltext}</div>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}
function base64ToPCMFloat32(base64) {
    try {
        const cleanBase64 = base64.replace(/[^A-Za-z0-9+/=]/g, '');
        const binary = atob(cleanBase64);
        let offset = 0;
        if (binary.length > 44 && binary.slice(0, 4) === 'RIFF') {
            offset = 44; // Skip WAV header
            console.log('Skipping WAV header (44 bytes)');
        }
        const pcmLength = binary.length - offset;
        const byteArray = new Uint8Array(pcmLength);
        
        for (let i = 0; i < pcmLength; i++) {
            byteArray[i] = binary.charCodeAt(i + offset);
        }
        const view = new DataView(byteArray.buffer);
        const sampleCount = byteArray.length / 2;
        const float32Array = new Float32Array(sampleCount);

        for (let i = 0; i < sampleCount; i++) {
            const int16 = view.getInt16(i * 2, true);
            float32Array[i] = int16 / 32768; // Convert to float32 (-1 to 1)
        }
        console.log(`Converted ${pcmLength} bytes to ${sampleCount} samples`);
        return float32Array;
    } catch (error) {
        console.error('Error in base64ToPCMFloat32:', error);
        return null;
    }
}
function playAudioChunk(base64audio) {
    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 44100});
            playheadTime = audioContext.currentTime;
        }
        
        const float32Arr = base64ToPCMFloat32(base64audio);
        console.log('Adding audio chunk with', float32Arr.length, 'samples to queue');
        
        const buffer = audioContext.createBuffer(1,float32Arr.length,44100);
        buffer.copyToChannel(float32Arr,0);

        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);

        const now = audioContext.currentTime;
        if(playheadTime< now + 0.15) playheadTime = now + 0.15;

        source.start(playheadTime);
        playheadTime += buffer.duration;

    } catch (error) {
        console.error("Error in playAudioChunk:", error);
        isPlayingAudio = false;
        if (!isRecording) {
            setTimeout(startRecording, 300);
        }
    }
}

window.addEventListener('beforeunload', stopRecording);