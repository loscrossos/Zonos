import sys
import os

if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"
elif sys.platform == "win32":
    os.environ['PHONEMIZER_ESPEAK_PATH'] = f'C:\\Program Files\\eSpeak NG'
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = f'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'



script_directory= os.path.dirname(__file__)
ai_model_base_dir=f"{script_directory}{os.sep}..{os.sep}ai_models{os.sep}"

os.environ['HF_HUB_CACHE'] = f'{ai_model_base_dir}hf_cache'
#old deprecated variable but seems to work more than the new one
os.environ['HUGGINGFACE_HUB_CACHE'] = f'{ai_model_base_dir}hf_cache'



AI_MODEL_DIR=f"{ai_model_base_dir}/tts/Zonos-v0.1-transformer/"

#AI_MODEL_DIR=f"{ai_model_base_dir}/tts/Zonos-v0.1-hybrid/"


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
torch.set_float32_matmul_precision('high')


text="well, well, well... look who is here.. the new mamba lib and its friends: causal and convid!"

cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
