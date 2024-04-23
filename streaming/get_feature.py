import torch
import torchaudio


class feature_extractor:
    def __init__(self, config):
        self.config = config
        self.feature_stack_n_frames = self.config.feature_stack_n_frames
        self.feature_stack_step = self.config.feature_stack_step
        self.transform = torch.nn.Sequential( 
        torchaudio.transforms.Preemphasis(0.97),  
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                n_fft=self.config.nfft,
                                            hop_length=self.config.hop_length,
                                            n_mels=self.config.n_mels ,
                                            center=False,
                                            mel_scale = 'slaney',
                                            norm='slaney',
                                            power=2,
        )

                )
    @staticmethod
    def stack_frames(x, n_frames=5, step=3):
        x = torch.stack([torch.flatten(x[i:i+n_frames, :]) for i in range(0, x.shape[0], step) if x[i:i+n_frames, :].shape[0]==n_frames], dim = 0)
        return x  
        
    
    def get_feature(self, signal):
        signal = torch.tensor(signal, requires_grad=False, dtype=torch.float32)
        feature = torch.log10(self.transform(signal).T + 1e-9)
        return feature
    