import json
from os.path import exists, dirname, abspath
import random
random.seed(12345)

root_projdir = dirname(dirname(abspath(__file__)))

splits = ['base', 'val', 'novel']
dataset_names = ['miniImagenet', 'tieredImagenet', 'cifar', 'CUB']

for split in splits:
    for dataset_name in dataset_names:
        root_images_dir = f'{root_projdir}/Datasets/{dataset_name}/Data'
        json_path = f'{root_projdir}/filelists/{dataset_name}/{split}_template.json'
        output_json_path = f'{root_projdir}/filelists/{dataset_name}/{split}.json'
        
        print(f'Generating {output_json_path}...')
        
        with open(json_path, "r") as read_file:
            json_dict = json.load(read_file)

        img_dir_final= []
        for img_addr in json_dict['image_names']:
            img_relpath = '/'.join(img_addr.split('/')[-2:])
            img_path = f'{root_images_dir}/{img_relpath}'
            # checking the existence of images only 1% of the time!
            if random.random() >= 0.99:
                assert exists(img_path), f'{img_path} does not exist'
            img_dir_final.append(img_path)
        json_dict['image_names']= img_dir_final
        
        with open(output_json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)
