import os
import pandas as pd
from tqdm import tqdm

def list_of_samples(path, save_path):
    splits = ['train', 'test', 'val']
    os.makedirs(save_path, exist_ok=True)
    for split in tqdm(splits):
        print(f'Processing {split} split')
        data = []
        for cls in os.listdir(f'{path}/{split}'):
            for img in os.listdir(f'{path}/{split}/{cls}'):
                data.append({'img': img, 'cls': cls, 'split': split})
        df = pd.DataFrame(data)
        df.to_csv(f'{save_path}/{split}.csv', index=False)
        print(f'Finished processing {split} split')

if __name__ == '__main__':
    path = '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Data/OCTID'
    save_path = '/home/marzieh/bigData/PrototypeBaseProject/PiPViT/annotations/OCTID'
    list_of_samples(path, save_path)