import os
import torch
import random
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    def __init__(self, data_path, data_txt_file):
        super().__init__()

        self.data_path = data_path
        self.frame_path = f'{data_path}/frames'

        with open(data_txt_file, 'r') as f:
            self.f_paths = f.readlines()

    def __len__(self):
            return len(self.f_paths)
    
    def __getitem__(self, i):
        '''
        Returns 2x src_files and target pairs from different videos
        '''
        src_1_f_path = self.f_paths[i].strip("\n")
        src_1 = torch.load(f'{self.frame_path}/{src_1_f_path}') / 255
        src_1_pose = torch.load(f'{self.data_path}/headposes/{src_1_f_path}') 
        # Randomly select drive video
        dir_name = src_1_f_path.split("/")[0]
        drive_paths = os.listdir(f'{self.frame_path}/{dir_name}')
        drive_paths.remove(src_1_f_path.split("/")[1])
        drive_1_path = random.choice(drive_paths)
        drive_1 = torch.load(f'{self.frame_path}/{dir_name.split("/")[0]}/{drive_1_path}') / 255
        drive_1_pose = torch.load(f'{self.data_path}/headpose/{dir_name.split("/")[0]}/{drive_1_path}')
        # Second pair should be of different video
        video_urls = os.listdir(self.data_path)
        video_urls.remove(f_path.split("/")[0])
        vid_url = random.choice(video_urls)
        src_2, drive_2 = random.choices(os.listdir(f'{self.data_path}/{vid_url}/{f_path}'), k=2)
        src_2 = torch.load(f'{self.frame_path}/{vid_url}/{f_path}/{src_2}') / 255
        src_2_pose = torch.load(f'{self.data_path}/headposes/{vid_url}/{f_path}/{src_2}') 
        drive_2 = torch.load(f'{self.frame_path}/{vid_url}/{f_path}/{drive_2}') / 255
        drive_2_pose = torch.load(f'{self.data_path}/headposes/{vid_url}/{f_path}/{src_2}') 

        return {
            'src_1': src_1,
            'src_1_pose': src_1_pose,
            'src_2': src_2,
            'src_2_pose': src_2_pose,
            'drive_1': drive_1,
            'drive_1_pose': drive_1_pose,
            'drive_2': drive_2,
            'drive_2_pose': drive_2_pose,
        }

    def collate_fn(self, batch):
        src_1, src_1_pose = [], []
        src_2, src_2_pose = [], []
        drive_1, drive_1_pose = [], []
        drive_2, drive_2_pose = [], []
        for data in batch:
            src_1.append(data['src_1'])
            src_1_pose.append(data['src_1_pose'])
            src_2.append(data['src_2'])
            src_2_pose.append(data['src_2_pose'])
            drive_1.append(data['drive_1'])
            drive_1_pose.append(data['drive_1_pose'])
            drive_2.append(data['drive_2'])
            drive_2_pose.append(data['drive_2_pose'])
        return {
            'src_1': torch.stack(src_1),
            'src_1_pose': src_1_pose,
            'src_2': torch.stack(src_2),
            'src_2_pose': src_2_pose,
            'drive_1': torch.stack(drive_1),
            'drive_1_pose': drive_1_pose,
            'drive_2': torch.stack(drive_2),
            'drive_2_pose': drive_2_pose,
        }


class HeadposeDataset(Dataset):
    def __init__(self, data_path, data_txt_file):
        super().__init__()

        self.frame_path = f'{data_path}/frames'
        self.headpose_path = f'{data_path}/headposes'

        with open(data_txt_file, 'r') as f:
            self.f_paths = f.readlines()

    def __len__(self):
            return len(self.f_paths)
    
    def __getitem__(self, i):
        f_path = self.f_paths[i].strip("\n")
        frame = torch.load(f'{self.frame_path}/{f_path}') / 255
        headpose = torch.load(f'{self.headpose_path}/{f_path}')
        headpose = torch.cat([headpose['euler'], headpose['translation_norm']])
        return {
            'frame':frame,
            'headpose': headpose,
        }

    def collate_fn(self, batch):
        frames, headposes = [], []
        for data in batch:
            frames.append(data['frame'])
            headposes.append(data['headpose'])
        return {
            'frames':torch.stack(frames),
            'headposes':torch.stack(headposes)
        }



def split_dataset(dataset, sizes):
    train_size = int(dataset.__len__() * sizes[0])
    val_size = dataset.__len__() - train_size
    train_dl, val_dl = random_split(dataset, [train_size, val_size])
    return train_dl, val_dl