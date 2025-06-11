import requests
import webbrowser
import os

def upload_to_tempsh(file_path):
    """Upload to temp.sh using correct API"""
    try:
        print(f"📤 Uploading {os.path.basename(file_path)} to temp.sh...")
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                'https://temp.sh/upload',
                files=files,
                timeout=30
            )
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
        if response.status_code == 200 and response.text.strip():
            url = response.text.strip()
            print(f"✅ Upload successful: {url}")
            webbrowser.open(url)
            return url
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Test it
if __name__ == "__main__":
    test_file = 'temp_clip.mp4'
    if os.path.isfile(test_file):
        upload_to_tempsh(test_file)
    else:
        print("File not found")
