import os
from huggingface_hub import snapshot_download

def download_whisper_model(target_path):
    """
    Downloads the whisper-large-v3-turbo model to a specific local directory.
    """
    # Create the directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    print(f"Starting download to: {target_path}")
    
    # Download the entire repository snapshot
    snapshot_download(
        repo_id="openai/whisper-large-v3-turbo",
        local_dir=target_path,
        local_dir_use_symlinks=False,  # Copies files directly instead of using symlinks
        revision="main"
    )
    
    print("Download complete! Files are ready at the location provided.")

if __name__ == "__main__":
    # Example: C:/Models/Whisper-Turbo or /home/user/models/whisper
    user_path = input("Enter the full path where you want to save the model: ").strip()
    download_whisper_model(user_path)
