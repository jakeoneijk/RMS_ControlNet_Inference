from typing import Union
import numpy as np
import librosa
from scipy.signal import firwin, lfilter, ellip, filtfilt
import torch

class RMS:
    @staticmethod
    def zero_phased_filter(x):
        b, a = ellip(4, 0.01, 120, 0.125) 
        x = filtfilt(b, a, x, method="gust")
        return x

    @staticmethod
    def get_rms(audio:np.ndarray, #[time]
                frame_lentgh:int,
                hop_length:int
                ) -> np.ndarray: #[time // hop_length]
        rms = librosa.feature.rms(y=audio, frame_length=frame_lentgh, hop_length=hop_length)
        rms = rms[0]
        rms = RMS.zero_phased_filter(rms)
        return rms
    
    @staticmethod
    def get_rms_fit_to_audio_ldm_mel(audio: Union[np.ndarray, torch.Tensor], #[time], [batch, time]
                                     frame_lentgh:int = 1024,
                                     hop_length:int = 160
                                     ) -> np.ndarray: #[time // hop_length]
        if isinstance(audio, np.ndarray): audio = torch.from_numpy(audio)
        if len(audio.shape) == 1: audio = audio.unsqueeze(0)
        audio = torch.nn.functional.pad( audio.float().unsqueeze(1), ( int((frame_lentgh - hop_length) / 2), int((frame_lentgh - hop_length) / 2), ), mode="reflect",)
        audio = audio.squeeze(1).detach().cpu().numpy()
        rms = librosa.feature.rms(y=audio, frame_length=frame_lentgh, hop_length=hop_length,center=False,pad_mode="reflect") #,normalized=False,onesided=True,
        rms = rms[0]
        rms = RMS.zero_phased_filter(rms)
        return rms.copy()
    
    @staticmethod
    def rms_distance(audio1: Union[np.ndarray, torch.Tensor], 
                     audio2: Union[np.ndarray, torch.Tensor]
                     ) -> float:
        assert len(audio1.shape) == 1 and len(audio2.shape) == 1, f"audio1.shape:{audio1.shape}, audio2.shape:{audio2.shape}"
        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]
        rms_audio1:np.ndarray = RMS.get_rms_fit_to_audio_ldm_mel(audio1)
        rms_audio2:np.ndarray = RMS.get_rms_fit_to_audio_ldm_mel(audio2)
        return float(np.mean(np.abs(rms_audio1 - rms_audio2)))
    
    @staticmethod
    def mu_law(rms:torch.Tensor, mu:int=255):
        '''Mu-law companding transformation'''
        # assert if all values of rms are non-negative
        assert torch.all(rms >= 0), f'All values of rms must be non-negative: {rms}'
        mu = torch.tensor(mu)
        mu_rms = torch.sign(rms) * torch.log(1 + mu * torch.abs(rms)) / torch.log(1 + mu)
        return mu_rms
    
    @staticmethod
    def inverse_mu_law(mu_rms:torch.Tensor, mu:int=255):
        '''Inverse mu-law companding transformation'''
        assert torch.all(mu_rms >= 0), f'All values of rms must be non-negative: {mu_rms}'
        mu = torch.tensor(mu)
        rms = torch.sign(mu_rms) * (torch.exp(mu_rms * torch.log(1 + mu)) - 1) / mu
        return rms
    
    @staticmethod
    def get_mu_bins(mu, num_bins, rms_min = 0.01):
        with torch.no_grad():
            mu_bins = torch.linspace(RMS.mu_law(torch.tensor(rms_min), mu), 1, steps=num_bins)
            mu_bins = RMS.inverse_mu_law(mu_bins, mu)
        return mu_bins
    
    @staticmethod
    def get_discretized_rms(rms, discretize_size:int, rms_min:float = 0.01,):
        mu_bins = RMS.get_mu_bins(discretize_size, discretize_size, rms_min)
        rms_discretized = RMS.discretize_rms(rms, mu_bins)
        rms_discretized = RMS.undiscretize_rms( rms_inds= rms_discretized, mu_bins = mu_bins)
        return rms_discretized
            
    
    @staticmethod
    def discretize_rms(rms, mu_bins):
        rms = torch.maximum(rms, torch.tensor(0.0)) # change negative values to zero
        rms_inds = torch.bucketize(rms, mu_bins, right=True) # discretize
        return rms_inds
    
    @staticmethod
    def undiscretize_rms(rms_inds, mu_bins, ignore_min=True):
        if ignore_min and mu_bins[0] > 0.0:
            mu_bins[0] = 0.0
        
        rms_inds_is_cuda = rms_inds.is_cuda
        if rms_inds_is_cuda:
            device = rms_inds.device
            rms_inds = rms_inds.detach().cpu()
        rms = mu_bins[rms_inds]
        if rms_inds_is_cuda:
            rms = rms.to(device)
        return rms