import queue
import threading
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import mlx_whisper
import os
import collections
import webrtcvad
import datetime

audio_queue = queue.Queue()

# Global session directory and ID dynamically generated at launch for immutably paired filenames
SESSION_ID = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
SESSION_DIR = "recordings"
os.makedirs(SESSION_DIR, exist_ok=True)

# Central accumulator for all VAD-isolated, mathematically cleansed human voice frames
session_audio_frames = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}", flush=True)
    audio_queue.put(indata.copy().flatten())

def append_session_log(text):
    """
    Saves finalized sentences into a paired HAM radio session log file.
    Uses UTC time formatting as per international amateur radio standards.
    """
    if not text.strip():
        return
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    filename = os.path.join(SESSION_DIR, f"session_{SESSION_ID}.log")
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text.strip()}\n")

class TextStabilizer:
    def __init__(self):
        self.history = collections.deque(maxlen=2)
        self.locked_words = []
        
    def step(self, raw_text):
        current_words = raw_text.split()
        self.history.append(current_words)
        
        newly_locked = []
        
        if len(self.history) == 2:
            common = self.history[0]
            for words in list(self.history)[1:]:
                idx = 0
                while idx < len(common) and idx < len(words) and common[idx].lower() == words[idx].lower():
                    idx += 1
                common = common[:idx]
                
            if len(common) > len(self.locked_words):
                new_words = common[len(self.locked_words):]
                self.locked_words.extend(new_words)
                newly_locked = new_words
                
        unstable_suffix = current_words[len(self.locked_words):]
        return " ".join(newly_locked), " ".join(unstable_suffix)
        
    def reset(self):
        self.history.clear()
        self.locked_words.clear()

def check_voice_webrtc(audio_float32, sample_rate, vad):
    pcm_data = (audio_float32 * 32767).astype(np.int16).tobytes()
    frame_length = int(sample_rate * 0.03) 
    bytes_per_frame = frame_length * 2
    
    speech_frames = 0
    total_frames = 0
    
    for i in range(0, len(pcm_data) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm_data[i:i + bytes_per_frame]
        try:
            if vad.is_speech(frame, sample_rate):
                speech_frames += 1
        except:
            pass
        total_frames += 1
        
    return (speech_frames / total_frames) > 0.1 if total_frames > 0 else False

def redraw_console(committed_text, current_unstable=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*65)
    print("🎙️ ASYNC HAM TRANSCRIPTION (WebRTC GMM & Prefix Matching)")
    print(f"Logging pure conversation to: recordings/session_{SESSION_ID}.wav")
    print("Listening... (Press Ctrl+C to stop & save session)")
    print("="*65 + "\n")
    
    for sentence in committed_text[-10:]:
        print(f"🗣️ {sentence}")
        
    if current_unstable:
        print(f"   ... {current_unstable} [🎙️...]")

def transcriber_worker(model_id, sample_rate):
    global session_audio_frames
    
    CHUNK_SEC = 0.5 
    pre_buffer_size = int(1.0 / CHUNK_SEC) 
    pre_buffer = collections.deque(maxlen=max(1, pre_buffer_size))
    
    audio_buffer = np.array([], dtype=np.float32)
    committed_text = []
    
    vad = webrtcvad.Vad(2) 
    stabilizer = TextStabilizer()
    
    nyquist = 0.5 * sample_rate
    filter_b, filter_a = signal.butter(5, [300.0 / nyquist, 3000.0 / nyquist], btype='band')
    
    is_speaking = False
    post_speech_timer = 0
    POST_SPEECH_SEC = 2.5 
    MAX_BUFFER_SEC = 12.0 

    redraw_console(committed_text, "")

    try:
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break 
            
            has_voice = check_voice_webrtc(audio_chunk, sample_rate, vad)
            
            if not is_speaking:
                if has_voice:
                    is_speaking = True
                    post_speech_timer = 0
                    committed_text.append("") 
                    
                    audio_buffer = np.array([], dtype=np.float32)
                    for past_chunk in pre_buffer:
                        audio_buffer = np.concatenate([audio_buffer, past_chunk])
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                else:
                    pre_buffer.append(audio_chunk)
            else:
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                if not has_voice:
                    post_speech_timer += CHUNK_SEC
                else:
                    post_speech_timer = 0 
                
                force_commit = (len(audio_buffer) >= (MAX_BUFFER_SEC * sample_rate))
                
                if post_speech_timer >= POST_SPEECH_SEC or force_commit:
                    clean_audio = signal.lfilter(filter_b, filter_a, audio_buffer).astype(np.float32)
                    
                    # Store the completely static-free, tightly bounded audio instance internally for Gemini export later!
                    session_audio_frames.append(clean_audio)
                    
                    result = mlx_whisper.transcribe(
                        clean_audio, 
                        path_or_hf_repo=model_id, 
                        fp16=True,
                        language="en"
                    )
                    
                    final_text = result.get("text", "").strip()
                    if committed_text:
                        committed_text[-1] = final_text
                    else:
                        committed_text.append(final_text)
                        
                    append_session_log(final_text)
                    
                    is_speaking = False
                    audio_buffer = np.array([], dtype=np.float32)
                    pre_buffer.clear()
                    stabilizer.reset()
                    
                    redraw_console(committed_text, "")
                else:
                    if audio_queue.qsize() > 1:
                        audio_queue.task_done()
                        continue
                        
                    clean_audio = signal.lfilter(filter_b, filter_a, audio_buffer).astype(np.float32)
                    
                    result = mlx_whisper.transcribe(
                        clean_audio, 
                        path_or_hf_repo=model_id, 
                        fp16=True,
                        language="en"
                    )
                    
                    raw_text = result.get("text", "").strip()
                    new_stable, unstable = stabilizer.step(raw_text)
                    
                    if new_stable and committed_text:
                        committed_text[-1] = committed_text[-1].strip() + " " + new_stable
                    
                    redraw_console(committed_text, unstable)

            audio_queue.task_done()
    except Exception as e:
        print(f"\nTranscriber worker encountered an error: {e}")

def main():
    FS = 16000 
    CHUNK_SEC = 0.5  
    BLOCK_SIZE = int(CHUNK_SEC * FS) 
    model_id = "/Users/apple/Downloads/AI_Models/whisper-large-v3-turbo"

    worker = threading.Thread(target=transcriber_worker, args=(model_id, FS), daemon=True)
    worker.start()

    try:
        with sd.InputStream(samplerate=FS, channels=1, dtype='float32',
                            blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n\nWrapping up HAM session... Exporting purified .wav file for Gemini.")
    finally:
        audio_queue.put(None)
        worker.join()
        
        if session_audio_frames:
            print("Combining VAD-verified audio blocks...")
            combined_audio = np.concatenate(session_audio_frames)
            
            # Export tightly packed float32 audio as proper PCM .wav to be perfectly compatible with Google's API!
            wav_path = os.path.join(SESSION_DIR, f"session_{SESSION_ID}.wav")
            wavfile.write(wav_path, FS, combined_audio)
            print(f"✅ Session successfully saved to {wav_path}")
            print(f"✅ Ready for Gemini Logging Plugin!")

if __name__ == '__main__':
    main()
