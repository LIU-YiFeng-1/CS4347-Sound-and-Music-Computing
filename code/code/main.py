
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import librosa
import numpy as np
from tqdm import tqdm

from model import BaseNN
from dataset import SingingDataset

import utils

FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)

class AST_Model:
    '''
        This is main class for training model and making predictions.
    '''
    def __init__(self, device= "cuda:0", model_path=None):
        # Initialize model
        self.device = device 
        self.model = BaseNN().to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded.')
        else:
            print('Model initialized.')

    def fit(self, args, learning_params):
        # Set paths
        trainset_path = args.train_dataset_path
        validset_path = args.valid_dataset_path
        save_model_dir = args.save_model_dir
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        # Load dataset
        print('Loading datasets...version_new')
        with open(trainset_path, 'rb') as f:
            trainset = pickle.load(f)
        with open(validset_path, 'rb') as f:
            validset = pickle.load(f)

        trainset_loader = DataLoader(
            trainset,
            batch_size=learning_params['batch_size'],
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        validset_loader = DataLoader(
            validset,
            batch_size=50,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Set optimizer and loss functions
        optimizer = optim.Adam(self.model.parameters(), lr=learning_params['lr'])
        onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0,], device=device))
        offset_criterion = nn.BCEWithLogitsLoss()
        octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)

        start_time = time.time()
        best_model_id = -1
        min_valid_loss = 10000
        epoch_num = learning_params['epoch']
        valid_every_k_epoch = learning_params['valid_freq']

        # Start training 
        print('Start training...')

        """ YOUR CODE HERE
        Hint: 
        1) Complete the training script including validation every n epoch. 
        2) Save the best model with the least validation loss 
        3) Return the epoch id with the best model after training.
        4) Printing out some necessary statistics, such as training loss of each sub-task, 
           may help you monitor the training process and understand it better.
        """
        best_model = 1000
        train_msg = ""
        validate_msg = ""
        for epoch in range(1, epoch_num+1):
            self.model.train()
            total_training_loss = 0
            total_training_split_loss = np.zeros(4)
            num_batches = 0
            validation_loss = 0
            validation_batch = 0

            for feature, label in trainset_loader:
                # preparing batch data
                print("Preparing feature and label in batches...")
                input_tensor = feature.to(self.device)
                label = label.to(self.device)
                onset_label = label[:, 0]
                offset_label = label[:, 1]
                octave_label = label[:, 2].type(torch.LongTensor).to(self.device)
                pitch_label = label[:, 3].type(torch.LongTensor).to(self.device)

                # print("\nOnset Label\n")
                # print(onset_label)
                # print("\nOffset Label\n")
                # print(offset_label)
                # print("\nOctave Label\n")
                # print(octave_label)
                # print("\nPitch Label\n")
                # print(pitch_label)

                # forward pass
                print("Forward passing...")
                prediction = self.model(input_tensor)

                # loss computing
                print("Computing losses...")
                onset_loss = onset_criterion(prediction[0], onset_label)
                offset_loss = offset_criterion(prediction[1], offset_label)
                octave_loss = octave_criterion(prediction[2], octave_label)
                pitch_loss = pitch_criterion(prediction[3], pitch_label)

                # backward pass
                print("Back propagation...")
                optimizer.zero_grad()
                onset_loss.backward(retain_graph=True)
                offset_loss.backward(retain_graph=True)
                octave_loss.backward(retain_graph=True)
                pitch_loss.backward()
                optimizer.step()

                # print loss
                print("Running losses...")
                num_batches += 1
                total_training_loss += onset_loss.detach().item() + offset_loss.detach().item() + octave_loss.detach().item() + pitch_loss.detach().item()
                print("Batch no.: " + str(num_batches))
                
            self.model.eval()

            with torch.no_grad():
                for feature, label in validset_loader:
                    # preparing batch data
                    print("Preparing validation set feature and label in batches...")
                    input_tensor = feature.to(self.device)
                    label = label.to(self.device)

                    onset_label = label[:, 0]
                    offset_label = label[:, 1]
                    octave_label = label[:, 2].type(torch.LongTensor).to(self.device)
                    pitch_label = label[:, 3].type(torch.LongTensor).to(self.device)

                    # forward pass
                    print("Forward passing the validation set...")
                    prediction = self.model(input_tensor)

                    # loss computing
                    print("Computing losses of the validation set...")
                    onset_loss = onset_criterion(prediction[0], onset_label)
                    offset_loss = offset_criterion(prediction[1], offset_label)
                    octave_loss = octave_criterion(prediction[2], octave_label)
                    pitch_loss = pitch_criterion(prediction[3], pitch_label)

                    print("Running losses of validation set...")
                    validation_loss += onset_loss.detach().item() + offset_loss.detach().item() + octave_loss.detach().item() + pitch_loss.detach().item()
                    validation_batch += 1
                    print("Validation batch no.: " + str(validation_batch))

            total_validation_loss = validation_loss/validation_batch
            total_loss = total_training_loss/num_batches

            elapsed_time = time.time() - start_time
            if epoch % 1 == 0 : 
                print('\nepoch=',epoch, '\t time=', elapsed_time,
                    '\t loss=', total_loss, '\t validation loss=', total_validation_loss)
            
            train_msg += "This is the no." + str(epoch) + " epoch with training loss = " + str(total_loss) + "(in " + str(elapsed_time) + ")"
            validate_msg += "This is the no." + str(epoch) + " epoch with validation loss = " + str(total_validation_loss) + "(in " + str(elapsed_time) + ")"
            
            if total_validation_loss < best_model:
                best_model = total_validation_loss
                best_model_id = epoch
                model_name = 'model_{}'.format(epoch)
                model_path = os.path.join(save_model_dir, model_name) 
                torch.save(self.model.state_dict(), model_path)
                print('\nBest model loss=' + str(best_model))
             
        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))
        print("this is the msg generated from the training loop: \n")
        print(train_msg)
        print("\nthis is the msg generated from the validation loop: \n")
        print(validate_msg)
        return best_model_id

    def parse_frame_info(self, frame_info, onset_thres, offset_thres):
        """Parse frame info [(onset_probs, offset_probs, note number)...] into desired label format."""

        result = []
        current_onset = None
        pitch_counter = []
        local_max_size = 3
        current_time = 0.0

        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i
            info = frame_info[i]

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

                if current_onset is None:
                    current_onset = current_time   
                else:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = current_time
                    pitch_counter = []

            # if it is offset
            elif info[1] >= offset_thres:  
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = None
                    pitch_counter = []

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                final_pitch = int(info[2]* 12 + info[3])
                if info[2] != 4 and info[3] != 12:
                    pitch_counter.append(final_pitch)

        if current_onset is not None:
            if len(pitch_counter) > 0:
                result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
            current_onset = None

        return result

    def predict(self, test_loader, results={}, onset_thres=0.1, offset_thres=0.5):
        """Predict results for a given test dataset."""

        # Start predicting
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}
            
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                song_ids = batch[1]

                result_tuple = self.model(input_tensor)
                onset_logits = result_tuple[0]
                offset_logits = result_tuple[1]
                pitch_octave_logits = result_tuple[2]
                pitch_class_logits = result_tuple[3]

                onset_probs, offset_probs = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()

                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    frame_info = (onset_probs[bid], offset_probs[bid], torch.argmax(pitch_octave_logits[bid]).item()
                            , torch.argmax(pitch_class_logits[bid]).item())

                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)    

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                results[song_id] = self.parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres)
        
        return results


if __name__ == '__main__':
    """
    This script performs training and validation of the singing transcription model.
    
    Sample usage:
    python main.py --train_dataset_path ./data/train.pkl --valid_dataset_path ./data/valid.pkl --save_model_dir ./results
    or 
    python main.py (All parameters are defualt)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', default='./data/train.pkl', help='path to the train set')
    parser.add_argument('--valid_dataset_path', default='./data/valid.pkl', help='path to the valid set')
    parser.add_argument('--save_model_dir', default='./results', help='path to save the trained models')
    
    args = parser.parse_args()
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ast_model = AST_Model(device)

    # Set learning params
    learning_params = {
        'batch_size': 50,
        'epoch': 5,
        'lr': 1e-4,
        'valid_freq': 1,
        'save_freq': 1
    }
    # Train and Validation
    best_model_id = ast_model.fit(args, learning_params)
    print("Best Model ID: ", best_model_id)
        

