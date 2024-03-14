from fastapi import FastAPI, File, UploadFile, Form
from faster_whisper import WhisperModel
from typing import List
import os
import tempfile

app = FastAPI()

model_size = "tiny.en"
model = WhisperModel(model_size, device="cuda", compute_type="float16")


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), save_path: str = Form(default=None)):
    # Use a temporary file to securely save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name

    # Transcribe the audio file
    segments, info = model.transcribe(temp_file_path, beam_size=5)

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Prepare the transcription text
    transcription = "\n".join([f"{segment.text}" for segment in segments])

    # Save transcription to a file if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(transcription)

    # Return the transcription as response
    response = {"transcription": transcription.split('\n')}
    return response
