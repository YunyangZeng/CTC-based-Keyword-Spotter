import pyaudio


class ListenerConfig:
    '''
        config for streaming audio from microphone
    '''
    def __init__(self):
        self.sample_rate = 16000 # sr
        self.format = pyaudio.paFloat32 # the format that pyaudio reads from the microphone
        self.channels = 1
        self.frames_per_buffer = 1024 # the chunk size that pyaudio reads from the microphone at a time


class AudioFileListenerConfig:
    '''
        config for streaming a pre-recorded audio file
    '''
    def __init__(self):
        self.sample_rate = 16000 # sr
        self.format = pyaudio.paFloat32 # the format that pyaudio reads from the audio file
        self.channels = 1
        self.frames_per_buffer = 1024 # the chunk size that pyaudio reads from the audio file at a time
        self.audio_file_path = r".\testvec\DE_Hey_Siri.wav"

class KeywordEnrollmentConfig:
    '''
        Config for keyword enrollment
        The large model is used for keyword enrollment
    '''
    def __init__(self):
        self.model_path = r'..\model_ckpts\large\model_epoch97.pth'
        self.sample_rate = 16000
        self.nfft = 512
        self.hop_length = 256
        self.n_mels = 23
        self.audio_keyword_path = r'.\keyword.wav'
        self.trascript_keyword_path = r'.\keyword.txt'
        self.decoder = 'beam' # decoder type, either 'greedy' or 'beam'
        self.feature_stack_n_frames = 5 # number of frames to stack
        self.feature_stack_step = 3  # step size to stack
        self.CMUdict_ARPAbet =  {"|":0,
                                "[SIL]": 1, "NG": 2, "F": 3, "M": 4, "AE": 5, "R": 6, "UW": 7, "N": 8, "IY": 9, "AW": 10, "V": 11, 
                                  "UH": 12, "OW": 13, "AA": 14, "ER": 15, "HH": 16, "Z": 17, "K": 18, "CH": 19, "W": 20, "EY": 21, 
                                  "ZH": 22, "T": 23, "EH": 24, "Y": 25, "AH": 26, "B": 27, "P": 28, "TH": 29, "DH": 30, "AO": 31, 
                                  "G": 32, "L": 33, "JH": 34, "OY": 35, "SH": 36, "D": 37, "AY": 38, "S": 39, "IH": 40, "[UNK]": 41, 
                                  "[PAD]": 42} # "|" is the blank token
        self.CMUdict_ARPAbet_inv = {v: k for k, v in self.CMUdict_ARPAbet.items()}

class RecorderConfig:
    '''
        Config for recording audio from microphone
    '''
    def __init__(self):
        self.sample_rate = 16000
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.frames_per_buffer = 1024 # the chunk size that pyaudio reads from the microphone at a time
        self.maximal_duration = 5 # records at maximum 5 seconds once started
        self.audio_keyword_path = r'.\keyword.wav' # the path to save the recorded audio


class DetectorConfig:
    '''
        Config for detecting keyword from streaming audio
        The lite model is used for keyword detection
    '''
    def __init__(self):
        self.model_path = r'..\model_ckpts\lite\model_epoch259.pth'  
        self.trascript_keyword_path = r'.\keyword.txt' # the keyword transcript path
        self.CMUdict_ARPAbet =  {"|":0,
                                "[SIL]": 1, "NG": 2, "F": 3, "M": 4, "AE": 5, "R": 6, "UW": 7, "N": 8, "IY": 9, "AW": 10, "V": 11, 
                                  "UH": 12, "OW": 13, "AA": 14, "ER": 15, "HH": 16, "Z": 17, "K": 18, "CH": 19, "W": 20, "EY": 21, 
                                  "ZH": 22, "T": 23, "EH": 24, "Y": 25, "AH": 26, "B": 27, "P": 28, "TH": 29, "DH": 30, "AO": 31, 
                                  "G": 32, "L": 33, "JH": 34, "OY": 35, "SH": 36, "D": 37, "AY": 38, "S": 39, "IH": 40, "[UNK]": 41, 
                                  "[PAD]": 42} # "|" is the blank token
        self.CMUdict_ARPAbet_inv = {v: k for k, v in self.CMUdict_ARPAbet.items()} # inverse dictionary

        # feature parameters
        self.sample_rate = 16000
        self.nfft = 512
        self.hop_length = 256
        self.n_mels = 23 
        self.feature_stack_n_frames = 5
        self.feature_stack_step = 3

        # audio buffer parameters
        self.audio_buffersize = int(self.sample_rate * 5)
        self.feature_buffersize = int((((self.audio_buffersize - self.nfft ) // self.hop_length + 1) - self.feature_stack_n_frames) // self.feature_stack_step + 1)  # number of frames in the feature buffer
        self.detection_window = int((((1.0 *self.sample_rate - self.nfft ) // self.hop_length + 1) - self.feature_stack_n_frames) // self.feature_stack_step + 1) # window size in frames for CTC forward score computation

        # detection parameters
        self.number_of_consecutive_hits = 1
        self.detection_step = int((0.5 * self.sample_rate // self.hop_length + 1) // self.feature_stack_step + 1) # step size in frames for CTC forward score computation
        self.skip_size = int((0.0 * self.sample_rate // self.hop_length + 1) // self.feature_stack_step + 1) # skip the next several frames to avoid repeated detection if a keyword is detected.
        self.confidence_threshold = 0.45 # confidence greater than this threshold is considered as a detection


        