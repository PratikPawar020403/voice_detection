import os
from huggingface_hub import HfApi

def push_to_huggingface():
    api = HfApi()
    repo_id = "pp22/voice-detection-api"
    repo_type = "space"

    print(f"Uploading to {repo_id}...")

    # Upload specific files to avoid hitting 1GB storage limit!
    files_to_upload = [
        "app.py", 
        "requirements.txt",
        "voice_detection_v2/voice_detector_neural.pt",
        "voice_detection_v2/config.json",
        "models/dsp_model_v2.pkl",
        "models/dsp_cols_v2.pkl"
    ]
    
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type=repo_type
            )

    # Upload only the src folder
    if os.path.exists("src"):
        print(f"Uploading folder: src...")
        api.upload_folder(
            folder_path="src",
            path_in_repo="src",
            repo_id=repo_id,
            repo_type=repo_type
        )

    print("\n✅ Upload complete! Your Space should be building now.")

if __name__ == "__main__":
    push_to_huggingface()
