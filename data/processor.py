import os
import torch
import shutil
import skvideo.io  
from tqdm import tqdm
from headpose.headpose import HeadposeInference

class Processor:
    def __init__(self, data_config):
        self.data_main_path = data_config["data_path"]
        self.headpose_path = f"{data_config['data_path']}/headposes"
        self.input_frames_path = f"{data_config['data_path']}/frames"
        self.input_videos_path = f"{data_config['video_dir']}"

    def __call__(self):
        self.extract_headposes()
        # self.extract_frames()
        # self.normalize_eulers()
        # self.create_headpose_train_file()

    def extract_headposes(self):
        headpose = HeadposeInference()
        for folder in tqdm(os.listdir(f'{self.input_frames_path}'), 'Headpose'):
            for file in os.listdir(f'{self.input_frames_path}/{folder}'):
                frame = torch.load(f'{self.input_frames_path}/{folder}/{file}').numpy()
                print(f'{self.input_frames_path}/{folder}/{file}')
                R, T = headpose.inference(frame)
                torch.save({'R': torch.tensor(R), 'T': torch.tensor(T)}, 
                            f'{self.headpose_path}/{folder}/{file}')

    def extract_frames(self):
        # Seperate frames
        for file in tqdm(os.listdir(f'{self.input_videos_path}'), 'Extracting frames'):
            input_video = f'{self.input_videos_path}/{file}'
            frames = skvideo.io.vread(input_video)
            shutil.rmtree(f'{self.input_frames_path}/{file.split(".")[0]}', ignore_errors=True)
            os.makedirs(f'{self.input_frames_path}/{file.split(".")[0]}', exist_ok=True)
            [torch.save(torch.tensor(frame), f'{self.input_frames_path}/{file.split(".")[0]}/{i}.pt') 
            for i, frame in enumerate(frames)]

    def normalize_eulers(self):
        # Normalise R & T matrices
        mean_T = torch.tensor([180,125])
        div_term = torch.tensor([20, 10])
        # for folder in tqdm(os.listdir(f'{self.headpose_path}'), 'Normalising R & T'):
            # for i, file in enumerate(os.listdir(f'{self.headpose_path}/{folder}')):
                # data = torch.load(f'{self.headpose_path}/{folder}/{i}.pt')
                # euler = data['euler']
                # translation = data['translation']
                # mean_T = (translation + mean_T) / 2
        for folder in tqdm(os.listdir(f'{self.headpose_path}'), 'Normalising R & T'):
            for i, file in enumerate(os.listdir(f'{self.headpose_path}/{folder}')):
                data = torch.load(f'{self.headpose_path}/{folder}/{i}.pt')
                print(data.keys())
                data['translation_norm'] = (data['translation'] - mean_T) / div_term
                torch.save(data, f'{self.headpose_path}/{folder}/{i}.pt')

    def create_headpose_train_file(self, create_base_train_file):
        files = []
        for folder in tqdm(os.listdir(f'{self.input_frames_path}'), 'Creating Train File'):
            for file in os.listdir(f'{self.input_frames_path}/{folder}'):
                files.append(f'{folder}/{file}\n')
        with open(f'{self.data_main_path}/data.txt', 'w') as f:
            f.writelines(files)
