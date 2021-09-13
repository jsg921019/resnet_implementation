import os
import numpy as np
import pickle
from torchvision.datasets import VisionDataset


class CIFAR10(VisionDataset):

    base_folder = 'cifar-10-batches-py'
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']
    meta_file = 'batches.meta'

    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)
        
        if not self._check_downloaded():
            print("Start download...")
            tarpath = self._download()
            print("Extract downloaded file...")
            self._extract(tarpath)
            print("Done!")

        self.train = train
        self.classes = self._get_classes()
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)} 
        self.data, self.targets = self._get_data()

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(image=img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img['image'], target

    def _get_classes(self):

        meta_path = os.path.join(self.root, self.base_folder, self.meta_file)
        with open(meta_path, 'rb') as meta_file:
            meta_data = pickle.load(meta_file, encoding='latin1')
        
        return meta_data['label_names']
       
    def _get_data(self):

        data = []
        targets = []
        
        if self.train:
            batch_list = self.train_list
        else:
            batch_list = self.test_list

        for file_name in batch_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                data.append(batch['data'])
                if 'labels' in batch:
                    targets.extend(batch['labels'])
                else:
                    targets.extend(batch['fine_labels'])
        
        data = np.vstack(data).reshape(-1, 3, 32, 32) # NCHW
        data = data.transpose((0, 2, 3, 1))  # NHWC
        data = (data - data.mean(0)) / 255.
        data = data.astype(np.float32)
    
        return data, targets

    def _check_downloaded(self):
        for fname in [*self.train_list, *self.test_list, self.meta_file]:
            fpath = os.path.join(self.root, self.base_folder, fname)
            if not os.path.isfile(fpath):
                return False            
        return True
    
    def _download(self):

        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        fname = os.path.basename(url)
        fpath = os.path.join(self.root, fname)
        
        if not os.path.isfile(fpath):

            from tqdm import tqdm
            import requests

            os.makedirs(self.root, exist_ok=True)
            r = requests.get(url, stream=True)

            with open(fpath, 'wb') as f:
                total_length = int(r.headers.get('content-length'))

                for chunk in tqdm(r.iter_content(chunk_size=1024), total=total_length/1024 + 1, unit_scale=1/1024, unit='Mb',
                                  bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n:.2f} Mb/{total:.2f} Mb [{elapsed}<{remaining}, {rate_fmt}{postfix}]"): 
                    if chunk:
                        f.write(chunk)

        return fpath

    def _extract(self, tarpath):
        
        import tarfile

        with tarfile.open(tarpath, 'r:gz') as tar:
            for tarinfo in tar:
                tar.extract(tarinfo, self.root)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")