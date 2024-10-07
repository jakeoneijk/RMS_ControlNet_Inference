import numpy as np
import soundfile as sf
import torch
import librosa

from AudioLDMControlNetInfer.AudioLDMControlNet import AudioLDMControlNet
from Util.RMS import RMS

rms_reference_audio_path:str = './test.wav'

audio, sr = librosa.load(rms_reference_audio_path, sr=16000)
audio:torch.Tensor = torch.from_numpy(audio).unsqueeze(0) #[1,time]
rms:torch.Tensor = torch.from_numpy(RMS.get_rms_fit_to_audio_ldm_mel(audio=audio)) #[1, time/hop]

audio_ldm_controlnet = AudioLDMControlNet(
    config_yaml_path = './AudioLDMControlNetInfer/ModelWeight/audioldm_original.yaml',
    control_net_pretrained_path = './AudioLDMControlNetInfer/ModelWeight/ControlNetstep300000.pth',
    vae_pretrained_path = './AudioLDMControlNetInfer/ModelWeight/vae_mel_16k_64bins.ckpt',
    vocoder_pretrained_path='./AudioLDMControlNetInfer/ModelWeight/hifigan_16k_64bins',
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)
generated_audio_by_text:np.ndarray = audio_ldm_controlnet.generate(
    text_prompt='people fart',
    rms=rms
)
sf.write('./generated_audio_by_text.wav', generated_audio_by_text, samplerate= 16000)

generated_audio_by_waveform:np.ndarray = audio_ldm_controlnet.generate(
    waveform=audio,
    rms=rms
)
sf.write('./generated_audio_by_waveform.wav', generated_audio_by_waveform, samplerate= 16000)