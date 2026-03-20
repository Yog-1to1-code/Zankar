import queue
import threading
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import mlx_whisper
import os
import collections
import webrtcvad

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}", flush=True)
    audio_queue.put(indata.copy().flatten())

class TextStabilizer:
    """
    DSA Prefix-Matching 'Local Agreement' Algorithm.
    Maintains a sliding window of previous transcriptions. 
    If a prefix of words remains entirely identical across 3 consecutive iterations, 
    it is mathematically 'stabilized' and can be permanently locked mid-sentence!
    """
    def __init__(self):
        self.history = collections.deque(maxlen=3)
        self.locked_words = []
        
    def step(self, raw_text):
        current_words = raw_text.split()
        self.history.append(current_words)
        
        newly_locked = []
        
        if len(self.history) == 3:
            # Find the longest common prefix across the last 3 transcript iterations
            common = self.history[0]
            for words in list(self.history)[1:]:
                idx = 0
                while idx < len(common) and idx < len(words) and common[idx].lower() == words[idx].lower():
                    idx += 1
                common = common[:idx]
                
            # Compare consensus against already locked words
            if len(common) > len(self.locked_words):
                new_words = common[len(self.locked_words):]
                self.locked_words.extend(new_words)
                newly_locked = new_words
                
        # The 'unstable' suffix is whatever words are actively being generated past the locked bounds
        unstable_suffix = current_words[len(self.locked_words):]
        return " ".join(newly_locked), " ".join(unstable_suffix)
        
    def reset(self):
        self.history.clear()
        self.locked_words.clear()

def check_voice_webrtc(audio_float32, sample_rate, vad):
    """
    Google WebRTC Gaussian Mixture Model (GMM) Voice Activity Detection.
    Immune to high-volume static, reacting purely to human voice frequencies.
    """
    pcm_data = (audio_float32 * 32767).astype(np.int16).tobytes()
    frame_length = int(sample_rate * 0.03) # 30ms frames
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
        
    # Activate if > 25% of audio frames contain verified human vocal cord data
    return (speech_frames / total_frames) > 0.25 if total_frames > 0 else False

def redraw_console(committed_text, current_unstable=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*65)
    print("🎙️ ASYNC HAM TRANSCRIPTION (WebRTC GMM & Prefix Matching)")
    print("Listening continuously... (Press Ctrl+C to stop)")
    print("="*65 + "\n")
    
    for sentence in committed_text[-10:]:
        print(f"🗣️ {sentence}")
        
    if current_unstable:
        print(f"   ... {current_unstable} [🎙️...]")

def transcriber_worker(model_id, sample_rate):
    CHUNK_SEC = 0.5 
    pre_buffer_size = int(1.0 / CHUNK_SEC) # 1 sec margin
    pre_buffer = collections.deque(maxlen=max(1, pre_buffer_size))
    
    audio_buffer = np.array([], dtype=np.float32)
    committed_text = []
    
    # 1. Initialize WebRTC VAD engine
    # Setting = 3 provides the absolute maximum aggression toward filtering background static
    vad = webrtcvad.Vad(3) 
    
    # 2. Initialize our DSA Streaming Integrator
    stabilizer = TextStabilizer()
    
    nyquist = 0.5 * sample_rate
    filter_b, filter_a = signal.butter(5, [300.0 / nyquist, 3000.0 / nyquist], btype='band')
    
    is_speaking = False
    post_speech_timer = 0
    POST_SPEECH_SEC = 1.2 
    MAX_BUFFER_SEC = 12.0 

    redraw_console(committed_text, "")

    try:
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break 
            
            # Use statistical GMM mapping instead of primitive volume threshold
            has_voice = check_voice_webrtc(audio_chunk, sample_rate, vad)
            
            if not is_speaking:
                if has_voice:
                    is_speaking = True
                    post_speech_timer = 0
                    committed_text.append("") # Drop a fresh empty line for the new speaker phrase!
                    
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
                    
                    result = mlx_whisper.transcribe(
                        clean_audio, 
                        path_or_hf_repo=model_id, 
                        fp16=True,
                        language="en"
                    )
                    
                    final_text = result.get("text", "").strip()
                    if committed_text:
                        # Overwrite the mid-stream stabilized text completely with the polished final Whisper result
                        committed_text[-1] = final_text
                    else:
                        committed_text.append(final_text)
                    
                    # Reset all variables for the next sentence
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
                    
                    # Pass the raw text through our DSA Prefix-Matching algorithm!
                    new_stable, unstable = stabilizer.step(raw_text)
                    
                    if new_stable and committed_text:
                        # Instantly append perfectly stabilized words onto the screen while they're still talking!
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
        pass
    finally:
        audio_queue.put(None)
        worker.join()

if __name__ == '__main__':
    main()
