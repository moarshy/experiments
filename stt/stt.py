import sounddevice as sd
import queue
import wave
import io
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI API client (ensure OPENAI_API_KEY is set in your environment)
openai_client = OpenAI()

# Global flag to control recording loop
stop_recording = False

# Audio settings
SAMPLE_RATE = 16000        # 16 kHz sampling rate (Whisper works well at 16 kHz)
CHUNK_DURATION = 3         # chunk length in seconds for each API call
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
TRANSCRIBE_MODEL = "gpt-4o-transcribe"

# Queue to hold audio chunks
audio_queue = queue.Queue()

# A BytesIO subclass to include a filename attribute.
class NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name

def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to collect audio chunks."""
    if status:
        print(f"Microphone status: {status}")
    # Store a copy of the audio chunk (as numpy array) into the queue
    audio_queue.put(indata.copy())

def transcribe_stream():
    """Generator function to stream transcription results from the microphone."""
    global stop_recording
    stop_recording = False
    full_transcript = ""
    # Start the input stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, 
        channels=1, 
        dtype='int16',
        blocksize=CHUNK_SIZE, 
        callback=audio_callback
    )
    stream.start()
    try:
        while True:
            # Break if stop is requested and no more audio chunks are left to process
            if stop_recording and audio_queue.empty():
                break
            try:
                # Retrieve the next audio chunk (non-blocking with timeout)
                audio_chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue  # No chunk ready yet; keep waiting.
            # Convert the audio chunk (numpy array) to WAV bytes for the API
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_chunk.tobytes())
            buf.seek(0)
            # Wrap the buffer in a NamedBytesIO so it has a proper filename
            named_buf = NamedBytesIO(buf.read(), "audio.wav")
            # Send the audio chunk to OpenAI Whisper API for transcription
            result = openai_client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, 
                file=named_buf
            )
            # Extract the transcribed text from the response
            chunk_text = result.text.strip()
            # Append the transcribed text to the full transcript
            if full_transcript:
                full_transcript += " " + chunk_text
            else:
                full_transcript = chunk_text
            # Yield the updated transcript (partial result) to update the UI
            yield full_transcript
    finally:
        # Stop and close the audio stream when done
        stream.stop()
        stream.close()

def stop_transcription():
    """Signal the recording loop to stop."""
    global stop_recording
    stop_recording = True

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Live Whisper Transcription\nClick **Start Transcription** to begin recording from your microphone.")
    transcript_box = gr.Textbox(label="Transcription", lines=5, placeholder="Transcribed text will appear here...")
    with gr.Row():
        start_btn = gr.Button("Start Transcription", variant="primary")
        stop_btn = gr.Button("Stop Transcription")
    # Bind the start/stop buttons to functions
    start_btn.click(fn=transcribe_stream, inputs=None, outputs=transcript_box)
    stop_btn.click(fn=stop_transcription, inputs=None, outputs=None)
    # Enable queue to allow streaming updates
    demo.queue()
# Launch the app
demo.launch()
