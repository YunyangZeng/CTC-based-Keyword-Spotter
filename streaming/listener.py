
import pyaudio
import numpy as np
import wave

class MicrophoneListener:
    def __init__(self, queue, config):
        self.config = config

        self.queue = queue
        self.format = self.config.format
        self.channels = self.config.channels
        self.rate = self.config.sample_rate
        self.frames_per_buffer = self.config.frames_per_buffer
        self.p = pyaudio.PyAudio()

    def callback(self, in_data, frame_count, time_info, status):
        in_data = np.frombuffer(in_data, dtype=np.float32)
        self.queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def listen(self):
        stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frames_per_buffer, stream_callback=self.callback)
        stream.start_stream()
        try:
            while stream.is_active():
                pass
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping microphone listener...")
        finally:
            self.queue.put(None)
            stream.stop_stream()
            stream.close()
            self.p.terminate()
            print("Microphone listener stopped.")

class AudioFileListener:
    def __init__(self, queue, config):
        self.config = config

        self.queue = queue
        self.format = self.config.format
        self.channels = self.config.channels
        self.rate = self.config.sample_rate
        self.frames_per_buffer = self.config.frames_per_buffer
        self.file_path = self.config.audio_file_path
        self.p = pyaudio.PyAudio()


    def listen(self):
        wf = wave.open(self.file_path, 'rb')
        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()), 
                             channels=wf.getnchannels(), 
                             rate=wf.getframerate(),
                             output=True,
                             frames_per_buffer=self.frames_per_buffer)
        
        format = np.int16  # Default format
        if wf.getsampwidth() == 2:
            format = np.int16
        elif wf.getsampwidth() == 4:
            format = np.float32


        data = wf.readframes(self.frames_per_buffer)
        try:
            while data != b'':
                data = np.frombuffer(data, dtype=format)
                if format is np.int16:
                    data = data.astype(np.float32) / np.iinfo(np.int16).max
            
                self.queue.put(data)
                data = wf.readframes(self.frames_per_buffer)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping microphone listener...")
        finally:
            wf.close()
            self.queue.put(None)
            stream.stop_stream()
            stream.close()
            self.p.terminate()
            print("Audio file listener stopped.")



