import requests

def upload_file(file_path):
    url = 'http://35.182.236.137:8000/transcribe'
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        response = requests.post(url, files=files)
        return response.text

file_path = '/home/ajeema/code/transcription-server/server/Faith.mp3'
response = upload_file(file_path)
print(response)
