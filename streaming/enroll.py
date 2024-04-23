
import time
import keyboard
from CTC import CTC
from get_feature import feature_extractor
from model import PRNetwork_large, PRNetwork_lite
import torch
import numpy as np
import soundfile as sf
from config import KeywordEnrollmentConfig
#from torchaudio.models.decoder import ctc_decoder



class KeywordEnrollment:
    def __init__(self, config):
        self.config = config
        self.blank = self.config.CMUdict_ARPAbet['|']
        self.CTC = CTC(blank = self.blank)
        self.feature_extractor = feature_extractor(self.config)
        self.model = PRNetwork_large()
        self.load_model_checkpoint(self.config.model_path)
        self.model.eval()         


    def load_model_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def remove_silence_front_and_end(self, seq):
        """
        Args:
            seq: (list):
                    list of phonemes
        Returns:
            seq: (list): 
                    list of phonemes with silence removed from the front and end
        """
        i = 0
        while seq[i] == self.config.CMUdict_ARPAbet['[SIL]']:
            i += 1
        j = len(seq)-1
        while seq[j] == self.config.CMUdict_ARPAbet['[SIL]']:
            j -= 1
        return seq[i:j+1]
    def remove_unks(self, seq):
        """
        Args:
            seq: (list):
                    list of phonemes
        Returns:
            seq: (list): 
                    list of phonemes with [UNK] removed
        """
        return [x for x in seq if x != self.config.CMUdict_ARPAbet['[UNK]']]
    def greedy_decode(self, logits):
        """
        Call the CTC greedy decoder
        Args:
            logits: (torch.Tensor) :
                    (1, T, V) output logits tensor of the model
        Returns:
            tokens: (list) :
                    list of list of phoneme tokens
            scores: (list) :
                    list of scores
        """
        tokens = self.CTC.ctc_greedy_decode(logits.squeeze().numpy(), ignore = [self.config.CMUdict_ARPAbet['[UNK]'], self.config.CMUdict_ARPAbet['[PAD]']])
        scores = [1]
        return tokens, scores
    
    def beam_decode(self, logits, beam_size = 5, n_best = 3):
        """
        Call the CTC prefix beam search decoder
        Args:
            logits: (torch.Tensor) :
                    (1, T, V) output logits tensor of the model
        Returns:
            tokens: (list) :
                    list of list of phoneme tokens
            scores: (list) :
                    list of scores
        """
        y = torch.nn.functional.softmax(logits, dim = -1)
        out = self.CTC.ctc_prefix_beam_search_decode(y.squeeze().numpy(), beam_size = beam_size, n_best = n_best, blank = self.blank)
        tokens = [x[0] for x in out]
        scores = [x[1] for x in out]
        return tokens, scores

    def predict(self, feat, decoder = 'greedy'):
        """
        Predict the phoneme sequence from the input feature with the model, using the specified decoder
        Args:
            feat: (torch.Tensor) :
                    (1, T, V) feature tensor of the keyword
            model: (torch.nn.Module) :
                    The Phoneme Recognition model
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(feat)
        if decoder == 'greedy':
            out = self.greedy_decode(logits)
        elif decoder == 'beam':
            out = self.beam_decode(logits)
        else:
            raise ValueError('Invalid decoder')
        return out
    
    def run(self):
        audio, _ = sf.read(self.config.audio_keyword_path, dtype='float32')
        feature = self.feature_extractor.get_feature(audio)
        if self.config.feature_stack_step > 1: 
            feature_stack = self.feature_extractor.stack_frames(feature, n_frames=self.config.feature_stack_n_frames, step=self.config.feature_stack_step).unsqueeze(0)
            feature = feature_stack
        else:
            feature = feature.unsqueeze(0)

        tokens, scores = self.predict(feature, decoder = self.config.decoder)
        scores = [x/2 for x in scores]
        scores = [np.exp(x)/np.sum(np.exp(scores)) for x in scores]

        print(tokens)
        print(scores)
        tokens = [self.remove_unks(self.remove_silence_front_and_end(token)) for token in tokens]
        decoded_phonemes_list = [[self.config.CMUdict_ARPAbet_inv[x] for x in token] for token in tokens]
        print('Keyword Recognized: ')
        print(decoded_phonemes_list )
        with open(self.config.trascript_keyword_path, 'w') as f:
            for i,j in zip(decoded_phonemes_list,scores) :
                f.write(r'%.4f: ' % j + ' '.join(i) + '\n')

if __name__ == '__main__':
    config = KeywordEnrollmentConfig()
    Enrollment = KeywordEnrollment(config)
    Enrollment.run()





    
        
        
        



    


