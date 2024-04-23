import torch
import numpy as np
from model import PRNetwork_lite
import matplotlib.pyplot as plt
from get_feature import feature_extractor
from CTC import CTC
import time
from datetime import timedelta
import math


class KeywordDetector:
    def __init__(self, config):
        self.config = config
        self.model = PRNetwork_lite()
        self.load_model_checkpoint(self.config.model_path)
        self.model.eval()
        self.ctc = CTC(blank = self.config.CMUdict_ARPAbet['|'])
        self.audio_buffer = np.zeros((self.config.audio_buffersize, ), dtype=np.float32)
        self.feature_extractor = feature_extractor(self.config)
        self.feature_buffer = np.zeros((self.config.feature_buffersize, self.config.n_mels * self.config.feature_stack_n_frames), dtype=np.float32)
        self.buffer_for_computing_featstack = np.zeros((self.config.feature_stack_n_frames + self.config.feature_stack_step , self.config.n_mels), dtype=np.float32)
        self.PR_prob_buffer = np.zeros((self.config.feature_buffersize, len(self.config.CMUdict_ARPAbet)), dtype=np.float32)
        self.target_keyword_transcripts = {} # key: transcript , value: score
        self.load_target_keyword_trascript()
        self.detection_counter = 0
        self.detection_window = self.config.detection_window
        self.detection_step = self.config.detection_step
        self.skip = False # skip the next several frames to avoid repeated detection if a keyword is detected.
        self.skip_counter = 0
        self.skip_size = self.config.skip_size
        self.hit_counter = 0
        self.nth_detection = 0

    def load_target_keyword_trascript(self):
        with open(self.config.trascript_keyword_path, 'r') as f:
            self.target_keyword_transcript_list = f.readlines()
        self.target_keyword_transcript_list = [x.strip().split() for x in self.target_keyword_transcript_list]
        self.target_keyword_transcripts = {float(x[0].strip(':')): self.ctc.standardize_label([self.config.CMUdict_ARPAbet[i] for i in x[1:]]) for x in self.target_keyword_transcript_list}

    def load_model_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def display_init(self):
        # display the input audio waveform in real-time
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.axes = [ax1, ax2]
        self.line1, = self.axes[0].plot(np.arange(0, self.config.audio_buffersize) , np.random.randn(self.config.audio_buffersize), color='yellow')
        self.line2, = self.axes[1].plot(np.arange(0, self.config.feature_buffersize) , np.random.randn(self.config.feature_buffersize), color='yellow')
        self.axes[0].set_facecolor('black')
        self.axes[1].set_facecolor('black')
        self.axes[0].set_ylim([-1.5, 1.5])
        self.axes[1].set_ylim([-42, 42])
        self.axes[0].set_xlim([0, self.config.audio_buffersize])
        self.axes[0].set_title('Real-time Audio Input', fontsize=16)
        self.axes[1].set_title('Real-time Phoneme Prediction', fontsize=16)
        self.axes[0].set_xlabel('Samples', fontsize=14)
        self.axes[0].set_ylabel('Amplitude', fontsize=14)
        self.axes[1].set_xlabel('frames', fontsize=14)
        self.axes[1].set_ylabel('Label', fontsize=14)
        self.axes[0].tick_params(labelsize=12, colors='white')
        self.axes[0].title.set_color('white')
        self.axes[1].tick_params(labelsize=12, colors='white')
        self.axes[1].title.set_color('white')
        self.axes[0].xaxis.label.set_color('white')
        self.axes[0].yaxis.label.set_color('white')
    def abort_display(self):
        plt.ioff()
        plt.close()


    def run(self, data, start_time):
        # Preprocess data, extract features, and perform model inference.
        self.audio_buffer = np.roll(self.audio_buffer, -len(data), axis = 0)
        self.audio_buffer[-len(data):] = data

        feature_raw = self.feature_extractor.get_feature(data)
        self.buffer_for_computing_featstack = np.roll(self.buffer_for_computing_featstack, -feature_raw.shape[0], axis = 0)
        self.buffer_for_computing_featstack[-feature_raw.shape[0]:, :] = feature_raw.numpy()
        feature = self.feature_extractor.stack_frames(torch.from_numpy(self.buffer_for_computing_featstack[:self.config.feature_stack_n_frames, ...]), 
                                                      n_frames=self.config.feature_stack_n_frames, step=self.config.feature_stack_step).unsqueeze(0)

        _, T, V = feature.shape
        if T != 1:
            print(T)
            raise ValueError('Feature stack shape error')
        
        self.feature_buffer = np.roll(self.feature_buffer, -T, axis = 0)
        self.feature_buffer[-T:, :] = feature.squeeze(0).numpy()
        with torch.no_grad():
            model_output = self.model.streaming_forward(feature)
            prob = torch.nn.functional.softmax(model_output, dim = -1).squeeze(0).numpy()
        self.PR_prob_buffer = np.roll(self.PR_prob_buffer, -prob.shape[0], axis = 0)
        self.PR_prob_buffer[-prob.shape[0]:,:] = prob

        if self.detection_counter >= self.detection_step and not self.skip:
            self.detection_counter = 0
            y = self.PR_prob_buffer[-self.config.detection_window:, :]
            confidence = 0.0
            for weight, transcript in self.target_keyword_transcripts.items():
                _, detection_window_total_prob = self.ctc.CTCforward(y, transcript)
                confidence += self.ctc.seg_length_normalization_and_ratio(y, detection_window_total_prob).round(5) * weight
            if math.isnan(confidence):
                confidence = 0.0
            if confidence >= self.config.confidence_threshold:
                self.hit_counter += 1
            else:
                self.hit_counter = 0
            print('Confidence: %4f'%confidence)
            if self.hit_counter >= self.config.number_of_consecutive_hits:
                elapsed = time.time() - start_time
                self.nth_detection += 1
                print(self.nth_detection, 'th detection.',' Keyword Detected at time: ', str(timedelta(seconds=elapsed)))
                self.hit_counter = 0
                self.model.reset_hidden()
                self.skip = True                
            
        if self.skip:
            self.skip_counter += T
            if self.skip_counter >= self.skip_size:
                self.skip = False
                self.skip_counter = 0

        self.detection_counter += T
        self.line1.set_ydata(self.audio_buffer)
        self.line2.set_ydata(self.PR_prob_buffer.argmax(axis = 1))

        plt.pause(0.001)
        plt.draw()


        
