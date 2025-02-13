import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import tensorflow as tf
import torch

machine = '14-501-10'
dataset = 'kerguelen2014'

#durs = [15, 299, 640]
#ovlps = [12.5, 149.5, 320]
PERCH_INPUT_SIZE = 160000
PERCH_OUTPUT_SIZE = 1280

DATA_PATH = '../../data/BlueFinLibrary/raw/'
ANNOT_PATH = 'annotations/'
ENCODER_PATH = '../../models/bird-vocalization-classifier/'
encoder = tf.saved_model.load(ENCODER_PATH)

LABEL_ENCODER = {'bma' : 1,
                 'bmb' : 2,
                 'bmz' : 3,
                 'abz' : 4,
                 'bmd' : 5,
                 'bpds' : 6,
                 'dcall' : 7,
                 'bp20hz' : 8,
                 'bp20plus': 9,
                 'bp' : 10}

LABELS = list(LABEL_ENCODER.keys())

if machine == '14-501-10':
    dur = 15
    ovlp = 12.5
elif machine == '14-501-11':
    dur = 299
    ovlp = 149.5
else:
    dur = 640
    ovlp = 320

file = f'final_dur{dur}_ovlp{ovlp}_trs0.5.csv'
#dur = int(re.search(r'dur(\d{2,3})', file).group(1))
#embed_dict = {}
#audio_dict = {}
sr = PERCH_INPUT_SIZE / dur

annot = pd.read_csv(os.path.join(ANNOT_PATH, file))
annot = annot[annot['dataset'] == dataset]

audio_list = []
embed_list = []
y_list = []


for idx in tqdm(range(len(annot))):
    dataset = annot.iloc[idx, annot.columns.get_loc('dataset')]
    filename = annot.iloc[idx, annot.columns.get_loc('filename')]
    audio_path = os.path.join(DATA_PATH, dataset, filename)
    start = annot.iloc[idx, annot.columns.get_loc('start_time')]


    audio, _ = librosa.load(path=audio_path + '.wav', sr=sr, offset=start, duration=dur)
    audio = audio[:PERCH_INPUT_SIZE]
    output = encoder.infer_tf(audio[np.newaxis, :])
    embed = torch.Tensor(output['embedding'].numpy())
    embed = embed.squeeze()
    embed_list.append(embed)
    audio_list.append(audio)
    y_list.append(torch.Tensor(np.array(annot.iloc[idx][LABELS], dtype='int32').flatten()))

#torch.save(embed_dict, os.path.join(machine, f'embed_dur{dur}_{n}.pt'))
#torch.save(audio_dict, os.path.join(machine, f'audio2_dur{dur}.pt'))
torch.save(embed_list, f'embed_{dur}_{dataset}.pt')
torch.save(audio_list, f'audio_{dur}_{dataset}.pt')
#torch.save(y_list, f'label_{dur}_{dataset}.pt')
print('job done!\n')


"""for subset in data2:
    for d, o in zip(durs, ovlps):
        print(f'Encoding subset={subset} dur={d} ovlp={o}...', end=' ')
        embed_dict = {}
        audio_dict = {}
        sr = PERCH_INPUT_SIZE / d

        annotation_file = os.path.join(ANNOT_PATH, subset, f'results_dur{d}_ovlp{o}_trs0.5.csv')

        if os.path.exists(annotation_file):
            annot = pd.read_csv(annotation_file)

            for name in list(np.unique(annot['filename'])):
                embed_dict[name] = {}
                audio_dict[name] = {}

            for idx in tqdm(range(len(annot))):
                dataset = annot.iloc[idx, annot.columns.get_loc('dataset')]
                filename = annot.iloc[idx, annot.columns.get_loc('filename')]
                audio_path = os.path.join(DATA_PATH, dataset, filename)
                start = annot.iloc[idx, annot.columns.get_loc('start_time')]
        
                audio, _ = librosa.load(path = audio_path + '.wav', sr=sr, offset=start, duration=d)
                audio = audio[:PERCH_INPUT_SIZE]
                output = encoder.infer_tf(audio[np.newaxis, :])
                embed = torch.Tensor(output['embedding'].numpy())
                embed = embed.squeeze()
                embed_dict[filename][start] = embed
                audio_dict[filename][start] = audio
        
            print(f'done! Saving embeddings and audio...', end=' ')
            torch.save(embed_dict, os.path.join(machine, f'embed_{subset}_dur{d}_ovlp{o}_trs0.5.pt'))
            torch.save(audio_dict, os.path.join(machine, f'audio_{subset}_dur{d}_ovlp{o}_trs0.5.pt'))
            print('done!\n')"""
