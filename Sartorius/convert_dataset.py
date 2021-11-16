from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools import mask as maskUtils
import json
base_dir = '/home/zhaohj/Documents/dataset/Kaggle/sartorius-cell-instance-segmentation'


def get_info(df):
    cat_ids = {name: id+1 for id, name in enumerate(df.cell_type.unique())}
    cats = [{'name': name, 'id': id} for name, id in cat_ids.items()]
    return cat_ids, cats


def coco_structure(df, cat_ids, cats):
    images = [{'id': id, 'width': row.width, 'height': row.height, 'file_name': f'train/{id}.png'}
              for id, row in df.groupby('id').agg('first').iterrows()]
    annotations = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mask = rle2mask(row['annotation'], row['width'], row['height'])
        c_rle = maskUtils.encode(mask)
        c_rle['counts'] = c_rle['counts'].decode('utf-8')
        area = maskUtils.area(c_rle).item()
        bbox = maskUtils.toBbox(c_rle).astype(int).tolist()
        annotation = {
            'segmentation': c_rle,
            'bbox': bbox,
            'area': area,
            'image_id': row['id'],
            'category_id': cat_ids[row['cell_type']],
            'iscrowd': 0,
            'id': idx
        }
        annotations.append(annotation)
    return {'categories': cats, 'images': images, 'annotations': annotations}


def rle2mask(rle, img_w, img_h):
    array = np.fromiter(rle.split(), dtype=np.uint)
    array = array.reshape((-1, 2)).T
    array[0] = array[0] - 1
    starts, lenghts = array
    mask_decompressed = np.concatenate(
        [np.arange(s, s + l, dtype=np.uint) for s, l in zip(starts, lenghts)])
    msk_img = np.zeros(img_w * img_h, dtype=np.uint8)
    msk_img[mask_decompressed] = 1
    msk_img = msk_img.reshape((img_h, img_w))
    msk_img = np.asfortranarray(msk_img)
    return msk_img


def run():
    df = pd.read_csv(f'{base_dir}/train.csv')
    cat_ids, cats = get_info(df)
    print(cats)
    # train_df, val_df = train_test_split(df, train_size=0.9, random_state=0)
    # train = coco_structure(train_df, cat_ids, cats)
    # val = coco_structure(val_df, cat_ids, cats)
    # with open(f'{base_dir}/COCO/train.json', 'w') as f:
    #     json.dump(train, f, ensure_ascii=True, indent=4)
    # with open(f'{base_dir}/COCO/val.json', 'w') as f:
    #     json.dump(val, f, ensure_ascii=True, indent=4)


run()
