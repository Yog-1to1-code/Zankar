import queue
import threading
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import mlx_whisper
import os
import collections

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}", flush=True)
    audio_queue.put(indata.copy().flatten())

def redraw_console(committed_text, current_text=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*60)
    print("🎙️ ASYNC HAM RADIO TRANSCRIPTION (DSP Filtered & Speed Optimized)")
    print("Listening continuously... (Press Ctrl+C to stop)")
    print("="*60 + "\n")
    
    for sentence in committed_text[-10:]:
        print(f"🗣️ {sentence}")
        
    if current_text:
        print(f"🗣️ {current_text} [🎙️...]")

def transcriber_worker(model_id, sample_rate):
    CHUNK_SEC = 0.5 # Gathers audio every 0.5s for faster feeling of response
    pre_buffer_sec = 1.0  
    pre_buffer_size = int(pre_buffer_sec / CHUNK_SEC)
    pre_buffer = collections.deque(maxlen=max(1, pre_buffer_size))
    
    audio_buffer = np.array([], dtype=np.float32)
    committed_text = []
    
    nyquist = 0.5 * sample_rate
    low = 300.0 / nyquist
    high = 3000.0 / nyquist
    try:
        filter_b, filter_a = signal.butter(5, [low, high], btype='band')
    except Exception as e:
        print("Scipy filter error:", e)
    
    is_speaking = False
    post_speech_timer = 0
    POST_SPEECH_SEC = 1.0 
    
    # Cap maximum buffer to 8 seconds to prevent inference times from blowing up
    MAX_BUFFER_SEC = 8.0 
    
    RMS_THRESHOLD = 0.005  

    redraw_console(committed_text, "")

    try:
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break 
            
            chunk_filtered = signal.lfilter(filter_b, filter_a, audio_chunk)
            rms = np.sqrt(np.mean(chunk_filtered**2))
            has_voice = rms > RMS_THRESHOLD
            
            if not is_speaking:
                if has_voice:
                    is_speaking = True
                    post_speech_timer = 0
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
                    
                    # 🚀 ENORMOUS SPEED BOOST: Disabling the language detection layer cuts inference time in half!
                    result = mlx_whisper.transcribe(
                        clean_audio, 
                        path_or_hf_repo=model_id, 
                        fp16=True,
                        language="en"
                    )
                    
                    text = result.get("text", "").strip()
                    if text:
                        committed_text.append(text)
                    
                    is_speaking = False
                    audio_buffer = np.array([], dtype=np.float32)
                    pre_buffer.clear()
                    
                    redraw_console(committed_text, "")
                else:
                    # Skip live preview processing ONLY if we are severely backed up
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
                    current_text = result.get("text", "").strip()
                    redraw_console(committed_text, current_text)

            audio_queue.task_done()
    except Exception as e:
        print(f"\nTranscriber worker encountered an error: {e}")

def main():
    FS = 16000 
    CHUNK_SEC = 0.5  
    BLOCK_SIZE = int(CHUNK_SEC * FS) 
    model_id = "/Users/apple/Downloads/AI_Models/whisper-large-v3-turbo"

    print(f"Loading MLX model from {model_id} ...")

    worker = threading.Thread(target=transcriber_worker, args=(model_id, FS), daemon=True)
    worker.start()

    try:
        with sd.InputStream(samplerate=FS, channels=1, dtype='float32',
                            blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\n\nLive transcription stopped by user.")
    finally:
        audio_queue.put(None)
        worker.join()

if __name__ == '__main__':
    main()
