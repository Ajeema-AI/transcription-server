from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from io import BytesIO
import soundfile as sf
import librosa
from pydub import AudioSegment
import numpy as np
import tempfile
import os

app = FastAPI()

class TranscriptionResponse(BaseModel):
    text: str

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-small.en"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

short_form_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device
)

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            transcription = await handle_audio(temp_file.name)
        return TranscriptionResponse(text=transcription)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def handle_audio(audio_path):
    song = AudioSegment.from_file(audio_path)
    chunks = [song[i:i + 240000] for i in range(0, len(song), 240000)]  # 4 minute chunks
    results = []
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as chunk_file:
            chunk.export(chunk_file.name, format="wav")
            audio_input, sr_original = sf.read(chunk_file.name, dtype='float32')
            if audio_input.ndim > 1:
                audio_input = np.mean(audio_input, axis=1)
            if sr_original != 16000:
                audio_input = librosa.resample(audio_input, orig_sr=sr_original, target_sr=16000)
            transcription = short_form_pipe(audio_input)['text']
            results.append(transcription)
    return " ".join(results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4)



#
#
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from pydantic import BaseModel
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from io import BytesIO
# import soundfile as sf
# import librosa
# from pydub import AudioSegment
# import numpy as np
#
# app = FastAPI()
#
#
# class TranscriptionResponse(BaseModel):
#     text: str
#
#
# # Setup CUDA device and data type
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#
# # Load the model and processor
# model_id = "distil-whisper/distil-small.en"
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# ).to(device)
# processor = AutoProcessor.from_pretrained(model_id)
#
# # Define ASR pipelines
# short_form_pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     torch_dtype=torch_dtype,
#     device=device
# )
#
#
# @app.post("/transcribe", response_model=TranscriptionResponse)
# async def transcribe_audio(file: UploadFile = File(...)):
#     try:
#         # Save temporary audio file
#         audio_data = await file.read()
#         audio_path = "temp_audio_file.mp3"
#         with open(audio_path, "wb") as f:
#             f.write(audio_data)
#
#         # Process the audio file
#         transcription = await handle_audio(audio_path)
#         return TranscriptionResponse(text=transcription)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# async def handle_audio(audio_path):
#     song = AudioSegment.from_mp3(audio_path)
#     chunks = [song[i:i + 240000] for i in range(0, len(song), 240000)]  # 4 minute chunks
#     results = []
#     for i, chunk in enumerate(chunks):
#         chunk.export(f"chunk_{i}.mp3", format="mp3")
#         audio_input, sr_original = sf.read(f"chunk_{i}.mp3", dtype='float32')
#
#         # Check if the audio is stereo and convert to mono by averaging the channels
#         if audio_input.ndim > 1:
#             audio_input = np.mean(audio_input, axis=1)
#
#         # Resample the audio if necessary
#         if sr_original != 16000:
#             audio_input = librosa.resample(audio_input, orig_sr=sr_original, target_sr=16000)
#
#         # Process the audio
#         transcription = short_form_pipe(audio_input)['text']
#         results.append(transcription)
#
#     return " ".join(results)
#
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4)
