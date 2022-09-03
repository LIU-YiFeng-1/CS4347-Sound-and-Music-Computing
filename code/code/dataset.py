from turtle import onscreenclick
import torch
from torch.utils.data import Dataset

import os
import json
import pickle
import librosa
import argparse
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def get_feature(y):
    '''
        This function computes Constant-Q Transform of the given signals.
    '''
    y = librosa.util.normalize(y)
    cqt_feature = np.abs(librosa.cqt(y, sr=44100, hop_length=1024, fmin=librosa.midi_to_hz(36), n_bins=84*2, bins_per_octave=12*2, filter_scale=1.0)).T
    return torch.tensor(cqt_feature, dtype=torch.float).unsqueeze(1)


class SingingDataset(Dataset):
    '''
        The is the main Dataset class that is used for preprocessing and preparing training and validation data. 
        Args:
            annotation_path: the path to the annotations
            data_dir: the directory path to the data 
    '''
    def __init__(self, annotation_path, data_dir):
        with open(annotation_path) as json_data:
            annotations = json.load(json_data)
        
        self.data = self.preprocess(data_dir, annotations)

    def preprocess(self, data_dir, annotations):
        data = []
        features = {}

        for the_dir in tqdm(os.listdir(data_dir)):
            wav_path = os.path.join(data_dir, the_dir, "Mixture.mp3")
            y, sr = librosa.core.load(wav_path, sr=None, mono=True)
            if sr != 44100:
                y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
            features[the_dir] = get_feature(y)

        for the_dir in tqdm(os.listdir(data_dir)):
            cqt_data = features[the_dir]
            annotation_data = annotations[the_dir]
            frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]

            label_data = self.get_labels(annotation_data, frame_num)
            zero_padding = torch.zeros((channel_num, cqt_size), dtype=torch.float)

            for frame_idx in range(frame_num):
                cqt_feature = []
                for frame_window_idx in range(frame_idx-5, frame_idx+6):
                    if frame_window_idx < 0 or frame_window_idx >= frame_num:
                        cqt_feature.append(zero_padding.unsqueeze(1))
                    else:
                        choosed_idx = frame_window_idx
                        cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

                cqt_feature = torch.cat(cqt_feature, dim=1)
                data.append((cqt_feature, label_data[frame_idx]))
        
        return data

    def get_labels(self, annotation_data, frame_num):
        # annotation format: [onset, offset, note_number]

        new_label = []

        cur_note = 0
        cur_note_onset = annotation_data[cur_note][0]
        cur_note_offset = annotation_data[cur_note][1]
        cur_note_number = annotation_data[cur_note][2]

        # note numbers ranging from C2 (36) to B5 (83)
        # octave class ranging from 0 to 4, octave class 0 to 3: octave 2 to 5, octave class 4: unknown class(silence)
        # pitch_class ranging from 0 to 12, pitch class 0 to 11: pitch C to B, pitch class 12: unknown class(silence)
        note_start = 36
        frame_size = 1024.0 / 44100.0
        oneset_count = 0
        offset_count = 0
        # label formart: [0/1, 0/1, octave_class_num, pitch_class_num]
        for i in range(frame_num):
            cur_time = i * frame_size
            label = [0, 0, 0, 0] # is_onset, is_offset, octave_class, pitch_class
            
            """ YOUR CODE HERE
            Hint: You need to consider four situations.
            1)For the silent frame 2) For the onset frame 3) For the offset frame 4) For the voiced frame
            """
            #to determine whether each frame contains the onset or offset of the note
            #divide the onset in sec by frame size to get frame id
            #from the json file the first song, the first singing voice note starts at 17 sec, ie before 
            #17s, can consider as silent frame ie [0,0,4,12]
            #frame num for example is 10364
            
            starting = int(cur_note_onset/frame_size)
            ending = int(cur_note_offset/frame_size)

            if i == starting:
                oneset_count += 1
                label[0] = 1
                label[1] = 0
                if cur_note_number in range(36,47):
                    label[2] = 0 # 2nd octave
                    label[3] = cur_note_number - note_start
                elif cur_note_number in range(48,59):
                    label[2] = 1 # 3rd octave
                    label[3] = cur_note_number - note_start - 12
                elif cur_note_number in range(60,71):
                    label[2] = 2 # 4th octave
                    label[3] = cur_note_number - note_start - 24
                elif cur_note_number in range(72,83):
                    label[2] = 3 # 5th octave
                    label[3] = cur_note_number - note_start - 36
                else:
                    label[2] = 4 # unknown (silence)
                    label[3] = 12 
                print("the onset count is: ")
                print(oneset_count)
                #print("\n" + str(i) + str(label))
                #print("\n" + str(cur_note))
            elif i > starting and i < ending:
                label[0] = 0
                label[1] = 0
                if cur_note_number in range(36,47):
                    label[2] = 0 # 2nd octave
                    label[3] = cur_note_number - note_start
                elif cur_note_number in range(48,59):
                    label[2] = 1 # 3rd octave
                    label[3] = cur_note_number - note_start - 12
                elif cur_note_number in range(60,71):
                    label[2] = 2 # 4th octave
                    label[3] = cur_note_number - note_start - 24
                elif cur_note_number in range(72,83):
                    label[2] = 3 # 5th octave
                    label[3] = cur_note_number - note_start - 36
                else:
                    label[2] = 4 # unknown (silence)
                    label[3] = 12 
                #print("\n" + str(i) + str(label))
                #print("\n" + str(cur_note))
            elif i == ending:
                offset_count += 1
                label[0] = 0
                label[1] = 1
                if cur_note_number in range(36,47):
                    label[2] = 0 # 2nd octave
                    label[3] = cur_note_number - note_start
                elif cur_note_number in range(48,59):
                    label[2] = 1 # 3rd octave
                    label[3] = cur_note_number - note_start - 12
                elif cur_note_number in range(60,71):
                    label[2] = 2 # 4th octave
                    label[3] = cur_note_number - note_start - 24
                elif cur_note_number in range(72,83):
                    label[2] = 3 # 5th octave
                    label[3] = cur_note_number - note_start - 36
                else:
                    label[2] = 4 # unknown (silence)
                    label[3] = 12 
                try:
                    cur_note += 1
                    cur_note_onset = annotation_data[cur_note][0]
                    cur_note_offset = annotation_data[cur_note][1]
                    cur_note_number = annotation_data[cur_note][2]
                except IndexError:
                    pass
                #print("\n" + str(i) + str(label))
                #print("\n" + str(cur_note))
                print("the offset count is: ")
                print(offset_count)
            else:
                label = [0,0,4,12]

            #print("\n" + str(i) + str(label))
            #print("\n" + str(cur_note))
            new_label.append(label)

        return np.array(new_label)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class OneSong(Dataset):
    '''
        The Dataset class is used for preprocessing and preparing testing data. 
        The difference is that this class is only used to prepare data of one song with song id and without annotations. 
        Args:
            input_path: the path to one song e.g. "./data/test/100/Mixture.mp3"
            song_id: id of the song e.g. 100 
    '''
    def __init__(self, input_path, song_id):
        y, sr = librosa.core.load(input_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
        y = librosa.util.normalize(y)
        
        self.data_instances = []
        cqt_data = get_feature(y)
        frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
        zeros_padding = torch.zeros((channel_num, cqt_size), dtype=torch.float)

        for frame_idx in range(frame_num):
            cqt_feature = []
            for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                # padding with zeros if needed
                if frame_window_idx < 0 or frame_window_idx >= frame_num:
                    cqt_feature.append(zeros_padding.unsqueeze(1))
                else:
                    choosed_idx = frame_window_idx
                    cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

            cqt_feature = torch.cat(cqt_feature, dim=1)
            self.data_instances.append((cqt_feature, song_id))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)

   
if __name__ == "__main__":
    """
    This script performs preprocessing raw data and prepares train/valid/test data.
    
    Sample usage:
    python dataset.py --debug  
    python dataset.py --data_dir ./data --annotation_path ./data/annotations.json --save_dataset_dir ./data/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/', help='path to the directory of data ')
    parser.add_argument('--annotation_path', default='./data/annotations.json', help='path to the annotations')
    parser.add_argument('--save_dataset_dir', default='./data/', help='path to save the generated dataset')
    parser.add_argument('--debug', default=False, action='store_true', help='True: process one example song, False: process train and valid sets')
    
    args = parser.parse_args()
    if args.debug == True:
        modes = ['example']
        annotation_path = './data/annotations_example.json'
    else:
        modes = ['train', 'valid']
        annotation_path = args.annotation_path

    for mode in modes:
        data_dir = os.path.join(args.data_dir, mode)
        dataset = SingingDataset(annotation_path=annotation_path, data_dir=data_dir)
        save_path = os.path.join(args.save_dataset_dir, mode+'.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset generated at {}.'.format(save_path))


    
