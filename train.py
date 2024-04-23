import torch
import torchaudio
import glob
import os
import textgrid
import numpy as np
from tqdm import tqdm
import soundfile as sf
from model import Network
from torchsummaryX import summary
from torchaudio.models.decoder import ctc_decoder
from utils import get_librispeech_data


class config():

    def __init__(self):
        self.model_dir = r"C:\Users\yunyangzeng\Documents\yunyangzeng\PR\PR_torch\model_lite_ckpts"
        self.num_epochs = 300
        self.batch_size = 80
        self.start_epoch = 1
        self.learning_rate = 1e-4
        self.feature_stack_n_frames = 5
        self.feature_stack_step = 3
        self.feat_padding_value = np.log10(1e-9)
        self.hidden_size = 128 # 128 for lite model, 512 for large model
        self.CMUdict_ARPAbet =  {"|":0,
                                "[SIL]": 1, "NG": 2, "F": 3, "M": 4, "AE": 5, "R": 6, "UW": 7, "N": 8, "IY": 9, "AW": 10, "V": 11, 
                                  "UH": 12, "OW": 13, "AA": 14, "ER": 15, "HH": 16, "Z": 17, "K": 18, "CH": 19, "W": 20, "EY": 21, 
                                  "ZH": 22, "T": 23, "EH": 24, "Y": 25, "AH": 26, "B": 27, "P": 28, "TH": 29, "DH": 30, "AO": 31, 
                                  "G": 32, "L": 33, "JH": 34, "OY": 35, "SH": 36, "D": 37, "AY": 38, "S": 39, "IH": 40, "[UNK]": 41, 
                                  "[PAD]": 42} # "|" is the blank token
        self.CMUdict_ARPAbet_unstress = { 'AA': 'AA', 'AE': 'AE', 'AH': 'AH', 'AO': 'AO', 'AW': 'AW', 'AY': 'AY', 'EH': 'EH', 'ER': 'ER', 'EY': 
    'EY', 'IH': 'IH', 'IY': 'IY', 'OW': 'OW', 'OY': 'OY', 'UH': 'UH', 'UW': 'UW', 
    'AA0': 'AA', 'AA1': 'AA', 'AA2': 'AA', 'AE0': 'AE', 'AE1': 'AE', 'AE2': 'AE', 'AH0': 'AH', 'AH1': 'AH', 'AH2': 'AH', 'AO0': 'AO', 'AO1': 'AO',
    'AO2': 'AO', 'AW0': 'AW', 'AW1': 'AW', 'AW2': 'AW', 'AY0': 'AY', 'AY1': 'AY', 'AY2': 'AY', 'B': 'B', 'CH': 'CH', 'D': 'D', 'DH': 'DH', 'EH0': 'EH',
    'EH1': 'EH', 'EH2': 'EH', 'ER0': 'ER', 'ER1': 'ER', 'ER2': 'ER', 'EY0': 'EY', 'EY1': 'EY', 'EY2': 'EY', 'F': 'F', 'G': 'G', 'HH': 'HH', 'IH0': 'IH',
    'IH1': 'IH', 'IH2': 'IH', 'IY0': 'IY', 'IY1': 'IY', 'IY2': 'IY', 'JH': 'JH', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'NG': 'NG', 'OW0': 'OW', 'OW1': 'OW',
    'OW2': 'OW', 'OY0': 'OY', 'OY1': 'OY', 'OY2': 'OY', 'P': 'P', 'R': 'R', 'S': 'S', 'SH': 'SH', 'T': 'T', 'TH': 'TH', 'UH0': 'UH', 'UH1': 'UH', 'UH2': 'UH',
    'UW0': 'UW', 'UW1': 'UW', 'UW2': 'UW', 'V': 'V', 'W': 'W', 'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH','spn': "[UNK]" ,'sp': "[UNK]", 'sil': '[SIL]', "[SIL]": "[SIL]", "|": "|", "": "[SIL]"
    } # LibriSpeech textgrid files have different stress levels, this dictionary is used to remove the stress levels.
        self.cv_audio_dir = r"C:\Users\yunyangzeng\Documents\yunyangzeng\PR\English\clips_16k"
        self.cv_textgrid_dir = r"C:\Users\yunyangzeng\Documents\yunyangzeng\PR\English\cv_textgrids\cv_textgrid_fixed_header"
        
        self.libri_audio_dir = r'C:\Users\yunyangzeng\Documents\yunyangzeng\PR\English\LibriSpeech\LibriSpeech_ASR_corpus'
        self.libri_textgrid_dir = r'C:\Users\yunyangzeng\Documents\yunyangzeng\PR\English\LibriSpeech\librispeech_alignments'
        
        self.train_data_ratio = 1.0
        self.criterion1_weight = 1.0
        self.criterion2_weight = 1.0
        self.beam_width = 20

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, config, audio_files, label_files): 
        self.orig_sr = 16000
        self.sr = 16000
        self.transform = torch.nn.Sequential( 
            torchaudio.transforms.Preemphasis(0.97),
            torchaudio.transforms.MelSpectrogram(sample_rate=self.sr,
                                                 n_fft=512,
                                                hop_length=256,
                                                n_mels=23,
                                                center=False,
                                                mel_scale = 'slaney',
                                                norm='slaney',
                                                power=2,
            )

                    )
        self.feat_padding_value = config.feat_padding_value
        self.audio_files = audio_files
        self.label_files = label_files
        self.CMUdict_ARPAbet =  config.CMUdict_ARPAbet
        self.CMUdict_ARPAbet_unstress = config.CMUdict_ARPAbet_unstress
        self.feature_stack_n_frames = config.feature_stack_n_frames
        self.feature_stack_step = config.feature_stack_step
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, ind):
        
        audio_item = self.audio_files[ind]
        label_item = self.label_files[ind]
        waveform, orig_sr = sf.read(audio_item) # load the audio file
        duration = len(waveform) / orig_sr

        return waveform, duration, label_item

    
    def textgrid_to_array(self, file, duration, frame_rate):
        if "librispeech" in file.lower():
            tg = textgrid.TextGrid.fromFile(file, False)[1]
        else:
            tg = textgrid.TextGrid.fromFile(file, False)[0]
        label_array = np.ones(int(np.ceil(duration * frame_rate))) * self.CMUdict_ARPAbet["[SIL]"] # create an array of length duration * sr and fill it with padding
        for i in tg:
            mark = self.CMUdict_ARPAbet_unstress[i.mark]
            label = self.CMUdict_ARPAbet[mark]
            mark_start = int(np.ceil(i.minTime * frame_rate))
            mark_end = int(np.ceil(i.maxTime * frame_rate))
            label_array[mark_start:mark_end] = label

        return label_array
    def textgrid_to_transcript(self, file): # for if using ctc loss
        if "librispeech" in file.lower():
            tg = textgrid.TextGrid.fromFile(file, False)[1]
        else:
            tg = textgrid.TextGrid.fromFile(file, False)[0]
        transcript = []
        for i in tg:
            mark = self.CMUdict_ARPAbet_unstress[i.mark]
            label = self.CMUdict_ARPAbet[mark]
            transcript.append(label)
        return transcript
    
    @staticmethod
    def collapse_repeats(label_array, ignore_token):
        collapsed_label_array = []
        previous = None
        for step_label in label_array:
            if step_label !=previous and step_label != ignore_token:
                collapsed_label_array.append(step_label)
            previous = step_label
        
        return collapsed_label_array      

    @staticmethod
    def stack_frames(x, n_frames=5, step=3):
        x = [torch.stack([torch.flatten(f[i:i+n_frames, :]) for i in range(0, f.shape[0], step) if f[i:i+n_frames, :].shape[0]==n_frames], dim = 0) for f in x]
        return x       
    
    def collate_fn(self,batch):

        batch_wave = [torch.from_numpy(data[0]) for data in batch]         
        batch_feat = [torch.log10(self.transform(waveform.float()).T + 1e-9) for waveform in batch_wave]       

        batch_trasncript = [torch.tensor(self.textgrid_to_transcript(data[2])) for data in batch]

        batch_feat = self.stack_frames(batch_feat, n_frames=self.feature_stack_n_frames, step=self.feature_stack_step)
        batch_feat_padded = torch.nn.utils.rnn.pad_sequence(batch_feat, batch_first=True, padding_value=self.feat_padding_value)

        lengths_batch_feat = torch.tensor([feat.shape[0] for feat in batch_feat])
        lengths_batch_transcript = torch.tensor([transcript.shape[0] for transcript in batch_trasncript])

        batch_transcript = torch.cat(batch_trasncript)
        return batch_feat_padded.float(), batch_transcript.long(), lengths_batch_feat, lengths_batch_transcript


def decat_sequence(sequence, lengths):
    decated = []
    start = 0
    for i, length in enumerate(lengths):
        end = start + length
        decated.append(sequence[start:end])
        start = end
    return decated

def calculate_PER(beam_out, targets):
    len_targets = [len(t) for t in targets]
    lev = [torchaudio.functional.edit_distance(beam_out[i][0].tokens, targets[i]) for i in range(len(beam_out))]
    lev = sum(lev) / sum(len_targets)

    return lev
        

def run_epoch(config, model, optimizer, criterion1, criterion2, loader, beam_search_decoder, device,  isTraining = True):
    epoch_ctc_loss = 0.0
    epoch_total_loss = 0.0
    epoch_correct_count = 0
    epoch_valid_frame_count = 0
    batch_bar = tqdm(loader, position=0, desc = 'Train' if isTraining else 'Valid', leave = True)
    epoch_levenshtein_distance = 0

    for i, batch in enumerate(tqdm(loader)):
        if isTraining:
            optimizer.zero_grad()
        x, t, l_x, l_t = batch
        x, t, l_x, l_t = x.to(device), t.to(device), l_x.cpu(), l_t.cpu()

        logits = model(x, l_x)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_ctc_loss = criterion2(log_probs.transpose(1, 0), t, l_x, l_t)

        epoch_ctc_loss += batch_ctc_loss.item() 

        batch_total_loss = batch_ctc_loss * config.criterion2_weight 

        if not isTraining:
            beam_out = beam_search_decoder(logits.clone().detach().cpu())
            decated_t = decat_sequence(t, l_t)
            batch_levenshtein_distance = calculate_PER(beam_out, decated_t) 
            epoch_levenshtein_distance += batch_levenshtein_distance


        if isTraining:
            batch_total_loss.backward()
            optimizer.step()

        if not isTraining:
            batch_bar.set_description("Testing ctc_loss: %f ,  PER: %f, %d/%d" % (epoch_ctc_loss/ (i+1),                                                                                     
                                                                          epoch_levenshtein_distance / (i+1), 
                                                                          i, 
                                                                          len(loader)))
        else:
            batch_bar.set_description("Training ctc_loss: %f  %d/%d" % (epoch_ctc_loss/ (i+1), 
                                                                         i,
                                                                         len(loader)))

    epoch_total_loss = epoch_ctc_loss / len(loader)
    epoch_levenshtein_distance /= len(loader)
    return epoch_total_loss, None, epoch_levenshtein_distance 
    

def train(config, model, optimizer, criterion1, criterion2, train_loader, device = 'cuda:0'):
    model.to(device)
    model.train()
    train_epoch_loss, train_epoch_accuracy, _ = run_epoch(config, model, optimizer, criterion1, criterion2, train_loader, None, device, isTraining = True)
    return train_epoch_loss, train_epoch_accuracy

def validate(config, model, criterion1, criterion2, valid_loader, decoder, device = 'cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        valid_epoch_loss, valid_epoch_accuracy, valid_levenshtein_distance = run_epoch(config, model, None, criterion1, criterion2, valid_loader, decoder, device, isTraining = False)
    return valid_epoch_loss, valid_epoch_accuracy, valid_levenshtein_distance

def save_state_dict(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, path)
def load_state_dict(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if not optimizer is None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass
    else:
        optimizer = None
    return model, optimizer, epoch, loss

def main():
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Device: ", DEVICE)
    my_config = config()
    cv_all_label_files = sorted(glob.glob(my_config.cv_textgrid_dir + "/*"), key=lambda name : int(name.split(".")[0].split("_")[-1]))
    cv_all_audio_files = [os.path.join(my_config.cv_audio_dir, i.replace(".TextGrid", ".wav").split("\\")[-1]) for i in cv_all_label_files]

    
    
    cv_train_label_files = cv_all_label_files[:int(len(cv_all_label_files)*my_config.train_data_ratio)]
    cv_train_audio_files = cv_all_audio_files[:int(len(cv_all_audio_files)*my_config.train_data_ratio)]
    cv_test_label_files = cv_all_label_files[int(len(cv_all_label_files)*my_config.train_data_ratio):]
    cv_test_audio_files = cv_all_audio_files[int(len(cv_all_audio_files)*my_config.train_data_ratio):]


    libri_train_audio_files, libri_train_label_files, libri_test_audio_files, libri_test_label_files = get_librispeech_data(my_config.libri_audio_dir, my_config.libri_textgrid_dir)

    train_audio_files = cv_train_audio_files + libri_train_audio_files
    train_label_files = cv_train_label_files + libri_train_label_files

    test_audio_files = libri_test_audio_files
    test_label_files = libri_test_label_files


    train_dataset = AudioDataset(my_config, train_audio_files, train_label_files)
    valid_dataset = AudioDataset(my_config, test_audio_files, test_label_files)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=my_config.batch_size, shuffle=True, num_workers=8,
            pin_memory=True, collate_fn=train_dataset.collate_fn
        )
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=my_config.batch_size, shuffle=False, num_workers=8,
            pin_memory=True, collate_fn=valid_dataset.collate_fn
        )
    
    train_beam_search_decoder = ctc_decoder(lexicon = None,
                                    tokens = list(my_config.CMUdict_ARPAbet.keys()),
                                   nbest = 1,
                                   beam_size = 1,
                                   blank_token = "|",
                                   sil_token = "[SIL]",
                                   unk_word = "[UNK]")

    val_beam_search_decoder = ctc_decoder(lexicon = None,
                                    tokens = list(my_config.CMUdict_ARPAbet.keys()),
                                   nbest = 1,
                                   beam_size = my_config.beam_width,
                                   blank_token = "|",
                                   sil_token = "[SIL]",
                                   unk_word = "[UNK]")
    
    torch.cuda.empty_cache()

    model = Network(hidden_size = my_config.hidden_size , output_size=len(train_dataset.CMUdict_ARPAbet)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=my_config.learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=my_config.learning_rate, momentum=0.9, weight_decay=1e-6)
    #criterion1 = torch.nn.CrossEntropyLoss(reduction = "sum" ,ignore_index = my_config.CMUdict_ARPAbet["[PAD]"])
    criterion2 = torch.nn.CTCLoss(reduction = "mean", zero_infinity=True, blank=my_config.CMUdict_ARPAbet["|"])

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.98)

    start_epoch = my_config.start_epoch

    start_accuracy = 0.0
    start_levenstein_distance = 100.0

    if os.path.exists(os.path.join(my_config.model_dir, "model_epoch{}.pth".format(start_epoch-1))):
        #model, optimizer, _, start_accuracy = load_state_dict(model, optimizer, os.path.join(my_config.model_dir, "model_epoch{}.pth".format(start_epoch-1)))
        model, optimizer, _, start_accuracy = load_state_dict(model, optimizer, os.path.join(my_config.model_dir, "model_epoch{}.pth".format(start_epoch-1)))
        print("Model loaded from epoch %d" % (start_epoch-1))
    else:
        print("No previous epoch model found, training from scratch.")
        start_epoch = 1
    for epoch in range(start_epoch, my_config.num_epochs + 1):
        print("=====================================Epoch %d=====================================" % epoch)
        train_loss, train_accuracy = train(my_config, model, optimizer, None, criterion2, train_loader, DEVICE)
        print("Train Epoch %d: Train loss: %f" % (epoch, train_loss))
        valid_loss, valid_accuracy, valid_levenshtein_distance = validate(my_config, model, None, criterion2, valid_loader, val_beam_search_decoder, 'cpu')
        print("Validation Epoch %d: Valid loss: %f, PER: %f" % (epoch, valid_loss, valid_levenshtein_distance))
        #if valid_levenshtein_distance < start_levenstein_distance:
        save_state_dict(model, optimizer, epoch, valid_loss, os.path.join(my_config.model_dir, "model_epoch{}.pth".format(epoch)))
            
        #    start_levenstein_distance = valid_levenshtein_distance
        print("Model saved")
        if scheduler.get_last_lr()[-1] > 1e-5:
            scheduler.step()


if __name__ == "__main__":
    main()