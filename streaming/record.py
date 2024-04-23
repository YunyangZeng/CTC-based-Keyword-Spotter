import pyaudio
import numpy as np
import time
import keyboard
import soundfile as sf
from config import RecorderConfig


class recorder:
    def __init__(self, config):
        self.config = config
        self.format = self.config.format
        self.channels = self.config.channels
        self.rate = self.config.sample_rate
        self.frames_per_buffer = self.config.frames_per_buffer
        self.audio = pyaudio.PyAudio()
        self.max_recording_length = self.config.maximal_duration * self.config.sample_rate
        self.input_buffer = np.zeros((self.max_recording_length, ), dtype=np.float32)

    def record(self):
        stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frames_per_buffer)
        try:
            print("Keyword Registration Starts ...")
            time.sleep(1)
            print("=" * 100)
            print("Press 'space' and say the keyword ..., Press 'esc' to exit ...")
            print("Waiting for SPACE key press ... \n")
            counter = 0
            while True:        
                if keyboard.is_pressed('esc'):
                    print('Exiting ... \n')
                    break
                elif keyboard.is_pressed('space'):
                    while keyboard.is_pressed('space') and counter <= self.max_recording_length:
                        data = stream.read(self.frames_per_buffer)
                        data = np.frombuffer(data, dtype=np.float32)
                        self.input_buffer = np.roll(self.input_buffer, -len(data), axis = 0)
                        self.input_buffer[-len(data):] = data
                        counter += 1
                    break
                else:
                    continue
            print("Recording finished, saving ... \n")
            sf.write(self.config.audio_keyword_path, self.input_buffer, self.config.sample_rate)
            print("Recording saved")
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping keyword register...")
        stream.stop_stream()
        stream.close()
        self.audio.terminate()


if __name__ == '__main__':
    config = RecorderConfig()
    rec = recorder(config)
    rec.record()