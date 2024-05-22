import numpy as np
import soundfile as sf
import torch
import librosa

from AudioLDMControlNetInfer.AudioLDMControlNet import AudioLDMControlNet
from Util.RMS import RMS

audio_path:str = './test.wav'


audio, sr = librosa.load(audio_path, sr=16000)
audio:torch.Tensor = torch.from_numpy(audio).unsqueeze(0) #[1,time]
rms:torch.Tensor = torch.from_numpy(RMS.get_rms_fit_to_audio_ldm_mel(audio=audio)) #[1, time/hop]
#discretized_rms:torch.Tensor = RMS.get_discretized_rms(rms, 128)

audio_ldm_controlnet = AudioLDMControlNet()
generated_audio_by_text:np.ndarray = audio_ldm_controlnet.generate(
    text_prompt='people fart',
    rms=rms
)

sf.write('./generated_audio_by_text.wav', generated_audio_by_text, samplerate= 16000)
'''
generated_audio_by_waveform:np.ndarray = audio_ldm_controlnet.generate(
    waveform=audio,
    rms=rms
)

sf.write('./generated_audio_by_waveform.wav', generated_audio_by_waveform, samplerate= 16000)
'''

print('finish')