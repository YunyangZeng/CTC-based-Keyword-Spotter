import textgrid
import os
import glob

def get_librispeech_data(wav_dir, textgrid_dir):
    textgrids = glob.glob(textgrid_dir + '/**/*.TextGrid', recursive=True)
    train_textgrids = []
    train_wavs = []
    test_textgrids = []
    test_wavs = []
    for idx, textgrid in enumerate(textgrids):
        if "train-clean-100" in textgrid:
            train_wavs.append(os.path.join( wav_dir, r'train-clean-100\LibriSpeech\train-clean-100', textgrid.split("train-clean-100\\")[1].replace(".TextGrid", ".flac")))
            train_textgrids.append(textgrid)
        elif "train-clean-360" in textgrid:
            train_wavs.append(os.path.join( wav_dir, r'train-clean-360\LibriSpeech\train-clean-360', textgrid.split("train-clean-360\\")[1].replace(".TextGrid", ".flac")))
            train_textgrids.append(textgrid)
        elif "train-other-500" in textgrid:
            train_wavs.append(os.path.join( wav_dir, r'train-other-500\LibriSpeech\train-other-500', textgrid.split("train-other-500\\")[1].replace(".TextGrid", ".flac")))
            train_textgrids.append(textgrid)
        elif "dev-clean" in textgrid:
            train_wavs.append(os.path.join( wav_dir, r'dev-clean\LibriSpeech\dev-clean', textgrid.split("dev-clean\\")[1].replace(".TextGrid", ".flac")))
            train_textgrids.append(textgrid)
        elif "dev-other" in textgrid:
            train_wavs.append(os.path.join( wav_dir, r'dev-other\LibriSpeech\dev-other', textgrid.split("dev-other\\")[1].replace(".TextGrid", ".flac")))
            train_textgrids.append(textgrid)
        elif "test-clean" in textgrid:
            test_wavs.append(os.path.join( wav_dir, r'test-clean\LibriSpeech\test-clean', textgrid.split("test-clean\\")[1].replace(".TextGrid", ".flac")))
            test_textgrids.append(textgrid)

    return train_wavs, train_textgrids, test_wavs, test_textgrids

if __name__ == "__main__":
    train_wavs, train_textgrids, test_wavs, test_textgrids = get_librispeech_data(r"C:\Users\AI3\Documents\yunyangzeng\Phoneme_Recognition\LibriSpeech\LibriSpeech_ASR_corpus",
                                r"C:\Users\AI3\Documents\yunyangzeng\Phoneme_Recognition\LibriSpeech\librispeech_alignments")
    print(test_textgrids[:10])
    print(test_wavs[:10])




