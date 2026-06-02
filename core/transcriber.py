import Whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")

_model = None

def load_model():
  global _model

  if _model is None:
    print("Loading Model..")
    _model = Whisper.load_model(WHISPER_MODEL)
    print("Whisper Model Loaded Successfully")

  return _model

def transcribed_chunk(chunk_path:str, translate:bool = False) -> str:
  model = load_model()

  task = "translate" if translate else "transcribe"

  result = model.transcribe(chunk_path,task = task)

  return result['text']

def transcribe_all(chunks : list, translate:bool = False) -> str:
  full_transcript = ""

  for i, chunk in enumerate(chunks):
    print(f"Trnascribing chunk {i+1}")

    text = transcribed_chunk(chunk, translate=translate)

    full_transcript += text + " "

  print("Transcription completed")

  return full_transcript
  