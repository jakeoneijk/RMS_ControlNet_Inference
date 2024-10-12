import numpy as np
import torch
import soundfile as sf
import librosa

from AudioLDMControlNetInfer.AudioLDMControlNet import AudioLDMControlNet
from Util.RMS import RMS
from TorchJaekwon.Util.UtilData import UtilData

rms_ref_audio_path:str = './Examples/rms_ref.wav'
sample_rate:int = 16000
duration_sec:float = 10.24

def load_audio(audio_path:str)->np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    audio:torch.Tensor = torch.from_numpy(audio).unsqueeze(0) #[1,time]
    audio = UtilData.fix_length(data=audio, length=int(sample_rate * duration_sec))
    return audio

rms_ref_audio:torch.Tensor = load_audio(rms_ref_audio_path)
rms:torch.Tensor = torch.from_numpy(RMS.get_rms_fit_to_audio_ldm_mel(audio=rms_ref_audio)) #[1, time/hop]

audio_ldm_controlnet = AudioLDMControlNet(
    control_net_pretrained_path = './ControlNetstep300000.pth',
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

generated_audio_by_text:np.ndarray = audio_ldm_controlnet.generate(
    text_prompt='dog bark loud',
    rms=rms
)
sf.write('./example_text_timb.wav', generated_audio_by_text, samplerate= sample_rate)

timb_ref_audio = load_audio('./Examples/timb_ref(footstep).mp3')
generated_audio_by_waveform:np.ndarray = audio_ldm_controlnet.generate(
    waveform=timb_ref_audio,
    rms=rms
)
sf.write('./example_audio_timb.wav', generated_audio_by_waveform, samplerate= sample_rate)