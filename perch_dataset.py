import os
import numpy as np
import pandas as pd
import librosa

import tensorflow as tf
import torch
from torch.utils.data import Dataset

class PerchDataset(Dataset):
    """
    Implementation of a Dataset returning embedding computed by the Perch model presented in
    Burooj Ghani, Tom Denton, Stefan Kahl, and Holger Klinck. 2023.
    Global birdsong embeddings enable superior transfer learning for bioacoustic classification.
    Sci Rep 13, 1 (December 2023), 22876. https://doi.org/10.1038/s41598-023-49989-z
    """

    def __init__(self, annotation_file, data_path, encoder_path, params, test_set, test_mode):
        self.all_annotations = pd.read_csv(annotation_file)
        self.data_path = data_path
        self.encoder = tf.saved_model.load(encoder_path)
        self.params = params

        if test_mode:
            self.annotations = self.all_annotations
        else:
            self.annotations = self.all_annotations[self.all_annotations['dataset'] != test_set]
        # TODO : doit y avoir moyen de faire un truc plus propre en suivant l'orga de Gabriel genre dans
        #  le dossier 'annotations' faire un sous-dossier pour chaque subset avec un train_val_annot et
        #  un test_annot et piocher l'un ou l'autre en fonction du test_set...
        #  Mais suppose de concat à chaque load, pas super opti

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        dur = self.params['dur'] if isinstance(self.params['dur'], (int, float)) else eval(self.params['dur'])
        sr = self.params['sr'] if isinstance(self.params['sr'], (int, float)) else eval(self.params['sr'])

        dataset = self.annotations.iloc[idx, self.annotations.columns.get_loc('dataset')]
        filename = self.annotations.iloc[idx, self.annotations.columns.get_loc('filename')]
        audio_path = os.path.join(self.data_path, dataset, filename)
        start = self.annotations.iloc[idx, self.annotations.columns.get_loc('start_time')]

        audio, _ = librosa.load(path=audio_path+'.wav', sr=sr, offset=start, duration=dur)
        audio = audio[:160000]
        output = self.encoder.infer_tf(audio[np.newaxis, :])
        embed = torch.Tensor(output['embedding'].numpy())
        embed = torch.squeeze(embed)

        idx_lab = [self.annotations.columns.get_loc(lab) for lab in self.params['labels']]
        labels = self.annotations.iloc[idx, idx_lab].to_numpy().astype(float)
        labels = torch.Tensor(labels)

        return (embed, labels)