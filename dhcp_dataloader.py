import torch
import os
import numpy as np
import pickle

def resize_pad(img):
    img_resi = np.pad(img, (30,30))
    img_resi = img_resi[:,17:273]
    
    return img_resi

class DHCP_2D(torch.utils.data.Dataset): 
    """2D MRI dhcp dataset loader"""

    def __init__(self, image_path='./dhcp-2d',
                 label_path='./labels_ants_full.pkl',
                 task='classification',
                 num_classes=2,
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = os.path.join(image_path)
        image_paths = sorted(os.listdir(self.img_dir))
        self.image_id = None
        file = open(os.path.join(label_path), 'rb')
        self.class_label = class_label
        self.labels = pickle.load(file)
        
        
        # choose if want preterm or term labels/images - filter images 
        if class_label == 1 :
           labels = self.labels[self.labels['is_prem'] == 1]
           self.labels = labels[labels['scan_ga'] <= 37]
        
        else:
            labels = self.labels[self.labels['is_prem'] == 0]
            self.labels = labels[labels['scan_ga'] > 37]
                    

        remove_ind = []
        i = 0
        # check which images are present in labels
        for img in image_paths:
            f = img.split('-')
            subject = 'CC' + f[0] 
            session = f[1].split('_')[0]
            
            if not (any(self.labels['id'].str.match(subject))):
                remove_ind.append(i)
            elif not (any(self.labels['session'] == int(session))):
                remove_ind.append(i)
            elif any(self.labels['id'].str.match(subject)):
                temp = self.labels.loc[self.labels['id'] == subject]
                if not (any((temp['session'] == int(session)).values)):
                    remove_ind.append(i)
            i = i + 1
            
        image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]


        self.image_paths = sorted(image_paths)
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, plot=False):
        
        # In the 2D case - load 3D image using nib
        img_name = sorted(self.image_paths)[idx]
        image = np.float32(np.load(os.path.join(self.img_dir, img_name)))
        
        if self.transform:
            image = self.transform(image)
            
        image = resize_pad(image) # added to resize image 256x256    
        # unsqueeze image to add 1st channel 
        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)
        
        # to get subject and session name 
        f = img_name.split('-')
        subject = 'CC' +  f[0]
        session = int(f[1].split('_')[0])


        if self.task == 'classification': # to classify image as preterm or term neonate
            values = self.labels.loc[self.labels['id'] == subject]
            values = values.loc[values['session'] == session]
            values = values['is_prem'].to_numpy().astype(int) 
            label = np.zeros(self.num_classes)
            label[values] = 1
            
        label = torch.from_numpy(label).float()
        
        return image

    