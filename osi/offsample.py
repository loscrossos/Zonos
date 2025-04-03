import sys
import os

if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"
elif sys.platform == "win32":
    os.environ['PHONEMIZER_ESPEAK_PATH'] = f'C:\\Program Files\\eSpeak NG'
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = f'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'




AI_MODEL_DIR="../../../ai_models/tts/Zonos-v0.1-transformer/"

AI_MODEL_DIR="../../../ai_models/tts/Zonos-v0.1-hybrid/"


config_path=f"{AI_MODEL_DIR}config.json"
model_path=f"{AI_MODEL_DIR}model.safetensors"




import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device




# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
#model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
model = Zonos.from_local(config_path, model_path, device=device)

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

cond_dict = make_cond_dict(text="Hello, world! did you know? WOW! i am SO excited to say this: this is the newest of the newest in mamba loading things!", speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
