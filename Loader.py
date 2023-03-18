# This is a train loader written by Mohsen Mozafari
# Train data and labes will be inside two folders in "main_dir". The data folder will be "data" and labels folder will be "labels"
# This is an example of how to use it:
#composed = torchvision.transforms.Compose([ToTensor(), Normalize()])

#main_dir = "D:\PHD\Thermal_21CNN\dummy_files"
#dataset = TrainLoader("D:\PHD\Thermal_21CNN\dummy_files", transform = composed)
#dataloader = DataLoader(dataset=dataset, batch_size = 4, shuffle=True)
import os
import numpy as np
import torch
import cv2

class TrainLoader():
    
    def __init__(self, main_dir, y, n, dim=(320,320), transform=None):
        data_dir = main_dir
        data_name = os.listdir(data_dir)
        self.data_name = data_name
        self.n_samples = len(data_name)
        self.data_dir = data_dir
        self.transform = transform
        self.y = y
        self.n = n
        self.dim = dim
    
    def __getitem__(self, index):
        loading_data_dir = os.path.join(self.data_dir, self.data_name[index])
        loaded_data = cv2.imread(loading_data_dir)
        loaded_data = cv2.resize(loaded_data, self.dim, interpolation = cv2.INTER_AREA)
        loaded_data = loaded_data/255.0
        loaded_data = (loaded_data - np.mean(loaded_data))/np.std(loaded_data)
        
        

        if self.data_name[index] in self.y:
                loaded_label = 1
        else:
                loaded_label = 0

        sample = np.array(loaded_data), loaded_label
        if self.transform:
            sample = self.transform(sample)
        return sample
            
    def __len__(self):
        return self.n_samples
    
class Normalize:
    def __call__(self, sample):
        inputs, targets = sample
        return inputs/255.0 , targets
    
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)