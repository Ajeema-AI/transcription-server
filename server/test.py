import requests


def transcribe_audio(file_path, save_path=None):
    url = 'http://localhost:8080/transcribe/'
    files = {'file': (file_path, open(file_path, 'rb'), 'audio/mpeg')}
    data = {'save_path': save_path} if save_path else {}

    response = requests.post(url, files=files, data=data)
    return response.json()


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    file_path = '/home/ajeema/code/WhisperLive/_Voicy_You_need_to_move_faster_than_that_son_speed_is_life_mp3.mp3'
    save_path = './transcription.txt'  # Server path where transcription will be saved. Set to None if not needed.

    transcription_response = transcribe_audio(file_path, save_path)
    print("Transcription:")
    for line in transcription_response["transcription"]:
        print(line)
