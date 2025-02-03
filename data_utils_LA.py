import torch
import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

# ASVFile = collections.namedtuple('ASVFile',
#     ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

# class ASVDataset(Dataset):
#     """ Utility class to load  train/dev datatsets """
#     def __init__(self, database_path=None,protocols_path=None,transform=None, 
#         is_train=True, sample_size=None, 
#         is_logical=True, feature_name=None, is_eval=False,
#         eval_part=0):

#         track = 'LA'   
#         data_root=protocols_path      
#         assert feature_name is not None, 'must provide feature name'
#         self.track = track
#         self.is_logical = is_logical
#         self.prefix = 'ASVspoof2019_{}'.format(track)
        
#         v1_suffix = ''
#         if is_eval and track == 'LA':
#             v1_suffix='_v1'
#             self.sysid_dict = {
#             '-': 0,  # bonafide speech
#             'A07': 1, 
#             'A08': 2, 
#             'A09': 3, 
#             'A10': 4, 
#             'A11': 5, 
#             'A12': 6,
#             'A13': 7, 
#             'A14': 8, 
#             'A15': 9, 
#             'A16': 10, 
#             'A17': 11, 
#             'A18': 12,
#             'A19': 13,
           
            
            
            
            
#         }
#         else:
#             self.sysid_dict = {
#             '-': 0,  # bonafide speech
            
#             'A01': 1, 
#             'A02': 2, 
#             'A03': 3, 
#             'A04': 4, 
#             'A05': 5, 
#             'A06': 6,
             
          
#         }

#         self.data_root_dir=database_path   
#         self.is_eval = is_eval
#         self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
#         print('sysid_dict_inv',self.sysid_dict_inv)

#         self.data_root = data_root
#         print('data_root',self.data_root)

#         self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
#         print('dset_name',self.dset_name)

#         self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
#         print('protocols_fname',self.protocols_fname)
        
#         self.protocols_dir = os.path.join(self.data_root)
#         print('protocols_dir',self.protocols_dir)
        
#         self.files_dir = os.path.join(self.data_root_dir, '{}_{}'.format(
#             self.prefix, self.dset_name ), 'flac')
#         print('files_dir',self.files_dir)

#         self.protocols_fname = os.path.join(self.protocols_dir,
#             'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
#         print('protocols_file',self.protocols_fname)

#         self.cache_fname = 'cache_{}_{}_{}.npy'.format(self.dset_name,track,feature_name)
#         print('cache_fname',self.cache_fname)
        
        
#         self.transform = transform

#         if os.path.exists(self.cache_fname):
#             self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
#             print('Dataset loaded from cache ', self.cache_fname)
#         else:
#             self.files_meta = self.parse_protocols_file(self.protocols_fname)
#             data = list(map(self.read_file, self.files_meta))
#             self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            
#             if self.transform:
#                 self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
#             torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            
#         if sample_size:
#             select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
#             self.files_meta= [self.files_meta[x] for x in select_idx]
#             self.data_x = [self.data_x[x] for x in select_idx]
#             self.data_y = [self.data_y[x] for x in select_idx]
#             self.data_sysid = [self.data_sysid[x] for x in select_idx]
            
#         self.length = len(self.data_x)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         x = self.data_x[idx]
#         y = self.data_y[idx]
#         return x, y, self.files_meta[idx]

#     def read_file(self, meta):
        
#         data_x, sample_rate = sf.read(meta.path)
#         data_y = meta.key
#         return data_x, float(data_y), meta.sys_id

#     def _parse_line(self, line):
#         tokens = line.strip().split(' ')
#         if self.is_eval:
#             return ASVFile(speaker_id=tokens[0],
#                 file_name=tokens[1],
#                 path=os.path.join(self.files_dir, tokens[1] + '.flac'),
#                 sys_id=self.sysid_dict[tokens[3]],
#                 key=int(tokens[4] == 'bonafide'))
#         return ASVFile(speaker_id=tokens[0],
#             file_name=tokens[1],
#             path=os.path.join(self.files_dir, tokens[1] + '.flac'),
#             sys_id=self.sysid_dict[tokens[3]],
#             key=int(tokens[4] == 'bonafide'))
        

   
#     def parse_protocols_file(self, protocols_fname):
#         lines = open(protocols_fname).readlines()
#         files_meta = map(self._parse_line, lines)
#         return list(files_meta)

class ASVDataset(Dataset):
    """
    Optimized dataset class for audio data with caching and parallel transformations.
    Compatible with CNN14 training, with padding for uniform input lengths.
    """
    def __init__(self, label_file, audio_dir, cache_file=None, n_jobs=-1, max_audio_length=64600):
        """
        Args:
            label_file (str): Path to the label file containing file mappings and labels.
            audio_dir (str): Path to the directory containing audio files.
            cache_file (str, optional): Path to save or load cached data.
            n_jobs (int, optional): Number of parallel jobs for processing.
            max_audio_length (int): Maximum length for audio padding.
        """
        self.label_file = label_file
        self.audio_dir = audio_dir
        self.cache_file = cache_file
        self.n_jobs = n_jobs
        self.max_audio_length = max_audio_length

        # If cache exists, load it
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            self.data_x, self.data_y, self.meta_data = torch.load(cache_file)
        else:
            print("Processing and loading data from source...")
            self.data_frame = self._parse_label_file(label_file)
            self.data_x, self.data_y, self.meta_data = self._load_and_process_data()

            # Save to cache if a cache file is specified
            if cache_file:
                print(f"Caching data to {cache_file}...")
                torch.save((self.data_x, self.data_y, self.meta_data), cache_file)

        self.length = len(self.data_x)

    def _parse_label_file(self, label_file):
        """
        Parse the label file to create a DataFrame with file names and labels.
        Args:
            label_file (str): Path to the label file.
        Returns:
            pd.DataFrame: DataFrame containing file names and labels.
        """
        data = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                file_name = parts[1] + '.flac'  # Use the second field as the file name
                label = parts[4]  # Use the fifth field as the label (bonafide or spoof)
                data.append({'file': file_name, 'label': label})
        return pd.DataFrame(data)

    def _load_and_process_data(self):
        """
        Load and preprocess audio files and labels.
        Returns:
            data_x: List of preloaded audio tensors.
            data_y: List of labels as tensors.
            meta_data: List of metadata dictionaries.
        """
        def load_audio(row):
            file_name = row['file']
            label = row['label']
            file_path = os.path.join(self.audio_dir, file_name)

            # Load audio file
            audio, sample_rate = sf.read(file_path)

            # Pad or truncate the audio to the fixed length
            audio_tensor = torch.zeros(self.max_audio_length, dtype=torch.float32)
            audio_length = min(len(audio), self.max_audio_length)
            audio_tensor[:audio_length] = torch.tensor(audio[:audio_length], dtype=torch.float32)

            # Metadata
            meta = {
                'file_name': file_name,
                'label': label,
                'sample_rate': sample_rate
            }
            return audio_tensor, int(label == 'bonafide'), meta

        # Use joblib for parallel processing
        results = Parallel(n_jobs=self.n_jobs)(delayed(load_audio)(row) for _, row in self.data_frame.iterrows())
        
        # Separate results
        data_x, data_y, meta_data = zip(*results)
        return list(data_x), list(data_y), list(meta_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get preloaded data and metadata by index.
        """
        audio = self.data_x[idx]
        label = self.data_y[idx]
        meta = self.meta_data[idx]
        return audio, label, meta
    
class CustomDataset(Dataset):
    """
    Optimized dataset class for ASV data.
    Audio files are preloaded and cached for efficiency.
    """

    def __init__(self, csv_file, audio_dir, transform=None, cache_file=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing file names and labels.
            audio_dir (str): Path to the directory containing audio files.
            transform (callable, optional): Transformations to apply to audio data.
            cache_file (str, optional): Path to save or load cached data.
        """
        self.csv_file = csv_file
        self.audio_dir = audio_dir
        self.transform = transform
        self.cache_file = cache_file

        # If cache exists, load it
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            self.data_x, self.data_y, self.meta_data = torch.load(cache_file)
        else:
            print("Processing and loading data from source...")
            self.data_frame = pd.read_csv(csv_file)
            self.data_x, self.data_y, self.meta_data = self._load_and_process_data()

            # Save to cache if a cache file is specified
            if cache_file:
                print(f"Caching data to {cache_file}...")
                torch.save((self.data_x, self.data_y, self.meta_data), cache_file)

        self.length = len(self.data_x)

    def _load_and_process_data(self):
        """
        Load and preprocess audio files and labels.
        Returns:
            data_x: List of preloaded audio tensors.
            data_y: List of labels as tensors.
            meta_data: List of metadata dictionaries.
        """
        data_x = []
        data_y = []
        meta_data = []

        # Helper function to load a single file
        def load_file(row):
            file_name = row['file']
            label = row['label']
            speaker = row['speaker']
            label_int = 1 if label == 'bona-fide' else 0
            file_path = os.path.join(self.audio_dir, file_name)
            audio, sample_rate = sf.read(file_path)

            # Apply transformation if provided
            if self.transform:
                audio = self.transform(audio)

            return audio, label_int, {'file_name': file_name, 'speaker': speaker}

        # Use joblib to parallelize the loading process
        results = Parallel(n_jobs=-1)(delayed(load_file)(row) for _, row in self.data_frame.iterrows())

        for audio, label, meta in results:
            data_x.append(audio)
            data_y.append(label)
            meta_data.append(meta)

        return data_x, data_y, meta_data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get preloaded data and metadata by index.
        """
        audio = self.data_x[idx]
        label = self.data_y[idx]
        meta = self.meta_data[idx]

        return audio, label, meta

class InTheWild(Dataset):
    """
    Optimized dataset class for audio data with caching and parallel transformations.
    Compatible with CNN14 training, with padding for uniform input lengths.
    """
    def __init__(self, csv_file, audio_dir, cache_file=None, n_jobs=-1, max_audio_length=64600):
        """
        Args:
            csv_file (str): Path to the CSV file containing file names and labels.
            audio_dir (str): Path to the directory containing audio files.
            cache_file (str, optional): Path to save or load cached data.
            n_jobs (int, optional): Number of parallel jobs for processing.
            max_audio_length (int): Maximum length for audio padding.
        """
        self.csv_file = csv_file
        self.audio_dir = audio_dir
        self.cache_file = cache_file
        self.n_jobs = n_jobs
        self.max_audio_length = max_audio_length

        # If cache exists, load it
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            self.data_x, self.data_y, self.meta_data = torch.load(cache_file)
        else:
            print("Processing and loading data from source...")
            self.data_frame = pd.read_csv(csv_file)
            self.data_x, self.data_y, self.meta_data = self._load_and_process_data()

            # Save to cache if a cache file is specified
            if cache_file:
                print(f"Caching data to {cache_file}...")
                torch.save((self.data_x, self.data_y, self.meta_data), cache_file)

        self.length = len(self.data_x)

    def _load_and_process_data(self):
        """
        Load and preprocess audio files and labels.
        Returns:
            data_x: List of preloaded audio tensors.
            data_y: List of labels as tensors.
            meta_data: List of metadata dictionaries.
        """
        def load_audio(row):
            file_name = row['file']
            label = row['label']
            file_path = os.path.join(self.audio_dir, file_name)

            # Load audio file
            audio, sample_rate = sf.read(file_path)

            # Pad or truncate the audio to the fixed length
            audio_tensor = torch.zeros(self.max_audio_length, dtype=torch.float32)
            audio_length = min(len(audio), self.max_audio_length)
            audio_tensor[:audio_length] = torch.tensor(audio[:audio_length], dtype=torch.float32)

            # Metadata
            meta = {
                'file_name': file_name,
                'label': label,
                'sample_rate': sample_rate
            }
            return audio_tensor, int(label == 'bona-fide'), meta

        # Use joblib for parallel processing
        results = Parallel(n_jobs=self.n_jobs)(delayed(load_audio)(row) for _, row in self.data_frame.iterrows())
        
        # Separate results
        data_x, data_y, meta_data = zip(*results)
        return list(data_x), list(data_y), list(meta_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get preloaded data and metadata by index.
        """
        audio = self.data_x[idx]
        label = self.data_y[idx]
        meta = self.meta_data[idx]

        return audio, label, meta