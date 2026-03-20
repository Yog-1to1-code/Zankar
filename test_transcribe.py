import mlx_whisper
import time

def main():
    # Local path to your downloaded model (MLX format)
    model_id = "/Users/apple/Downloads/AI_Models/whisper-large-v3-turbo"
    
    print(f"Loading MLX model from {model_id}...")
    
    # -------------------------------------------------------------------------
    # REPLACE the string below with the path to an actual audio file you have.
    # -------------------------------------------------------------------------
    audio_file = "sample2.mp3" 
    
    print(f"\nAttempting to transcribe: {audio_file}")
    start_time = time.time()
    
    try:
        # mlx_whisper automatically uses the GPU on Apple Silicon!
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=model_id,
            fp16=True
        )
        
        print("\n--- Transcription Result ---")
        print(result["text"])
        print("----------------------------")
        print(f"Transcription took {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"\nAn error occurred during transcription: {e}")

if __name__ == '__main__':
    main()
