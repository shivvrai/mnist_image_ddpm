import os
import sys
from huggingface_hub import HfApi, login

def deploy_code():
    print("Welcome! This will deploy your latest backend code to your Hugging Face Space.")
    print("-----------------------------------------------------------------------------")
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Please paste your Hugging Face Access Token (with 'Write' permissions) and press Enter: ").strip()
    
    if not token:
        print("Error: No token provided.")
        sys.exit(1)
        
    try:
        login(token=token)
        api = HfApi()
    except Exception as e:
        print(f"Failed to authenticate: {e}")
        sys.exit(1)
        
    # The user's actual Space repo for the app
    repo_id = "shivv01/mnist-backend"
    
    print(f"\nDeploying the latest code to Space '{repo_id}'...")
    try:
        api.upload_folder(
            folder_path="backend",
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy updated app.py with HF Hub download logic",
            ignore_patterns=["__pycache__/*", "*.pyc", "Dockerfile", "scratch_test.py", "train_classifier.py", "*.h5", "*.keras", "schemas.py"]
        )
        print("Deploy complete! Your Space is now restarting with the new code.")
    except Exception as e:
        print(f"Failed to deploy code: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy_code()
