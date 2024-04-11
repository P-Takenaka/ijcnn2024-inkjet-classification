import argparse
import functools
import pickle
import os
from multiprocessing import Pool
import time

import numpy as np
import pandas as pd

from PIL import Image

from src.feature_extraction import process_segment, get_features

Image.MAX_IMAGE_PIXELS = 1000000000

parser = argparse.ArgumentParser()

parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--num_crops_per_image', type=int, default=96)

parser.add_argument('--random_seed', type=int, default=1)

def run():
    args = parser.parse_args()

    output_file = './extracted_features.pkl'

    with open('metadata.pkl', 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data['scan_data'])

    paper_printer_models = [
        {
        'name': 'Canon - PIXMA MP280',
        'files': [
            'canon_pixma-mp280_best_photo.png',
            'canon_pixma-mp280_normal_photo.png',
        ]
        },

        {
        'name': 'Canon - PIXMA MG5350',
        'files': [
            'Canon_MG5350_Photo_Standard.png',
            'Canon_MG5350_Normalpaper_Standard.png',
        ]
        },

        {
        'name': 'Canon - PIXMA MX870',
        'files': [
            'Canon_MX870_Normalpaper.png',
            'Canon_MX870_Photo_Standard.png',
        ]
        },

        {
        'name': 'Canon - PIXMA Pro9500MK2',
        'files': [
            'canon_pro-9500ii_normal_photo.png',
            'canon_pro-9500ii_high_photo.png',
        ]
        },

        {
        'name': 'Canon - G3560',
        'files': [
            'canon_g3560_unknown_normal.png',
            'canon_g3560_unknown_normal_2.png',
        ]
        },

        {
        'name': 'Canon - TS8350',
        'files': [
            'canon_ts8350_normal_normal.png',
            'canon_ts8350_normal_normal_3.png',
        ]
        },

        {
        'name': 'Canon - G4511',
        'files': [
            'canon_g4511_unknown_normal.png',
            'canon_g4511_unknown_normal_2.png',
        ]
        },

        {
        'name': 'Canon - GX6050',
        'files': [
            'canon_gx6050_normal_normal_2.png',
            'canon_gx6050_high_normal.png',
        ]
        },

        {
        'name': 'Brother - MFC-825DW',
        'files': [
            'Brother_MFC-825DW_Photo_Standard.png',
            'Brother_MFC-825DW_Photo_High.png',
        ]
        },

        {
        'name': 'Brother - MFC-J6710DW',
        'files': [
            'brother_mfc-J6710dw_normal_photo.png',
            'brother_mfc-J6710dw_fast_photo_2.png',
        ]
        },

        {
        'name': 'Brother - DCP-J715W',
        'files': [
            'brother_dcpj715w_normal_photo_2.png',
            'brother_dcpj715w_best_photo.png',
        ]
        },

        {
        'name': 'Brother - MFC-265W',
        'files': [
            'brother_mfc-j265w_unknown_photo.png',
            'brother_mfc-j265w_high_photo.png',
        ]
        },

        {
        'name': 'HP - Deskjet 5652',
        'files': [
            'hp_5652_best_photo.png',
            'hp_5652_normal_photo.png',
        ]
        },

        {
        'name': 'HP - Deskjet 6122',
        'files': [
            'hp_deskjet-6122_high_normal.png',
            'hp_deskjet-6122_normal_normal.png',
        ]
        },

        {
        'name': 'HP - PSC 1210',
        'files': [
            'hp_psc-1210_best_normal.png',
            'hp_psc-1210_normal_normal.png',
        ]
        },

        {
        'name': 'HP - PSC 2175',
        'files': [
            'HP_PSC2175_Foto_Normal_011.png',
            'HP_PSC2175_Photo_High.png',
        ]
        },

        {
        'name': 'HP - Photosmart C4280',
        'files': [
            'hp_c4280_best_normal.png',
            'hp_c4280_normal_normal.png',
        ]
        },

        {
        'name': 'HP - X451DW',
        'files': [
            'hp_x451dw_normal_photo.png',
            'hp_x451dw_normal_normal.png',
        ]
        },

        {
        'name': 'Epson - ET-2650',
        'files': [
            'Epson_ET-2650_Foto_Standard.png',
            'Epson_ET-2650_Normalpaper.png',
        ]
        },

        {
        'name': 'Epson - WF-2835',
        'files': [
            'Epson_WF-2835_Photo_High.png',
            'Epson_WF-2835_Foto_Standard.png',
        ]
        },

        {
        'name': 'Epson - WF-C579R',
        'files': [
            'Epson_WF-C579R_Photo_Standard.png',
            'Epson_WF-C579R_Photo_High.png',
        ]
        },

        {
        'name': 'Epson - XP-352',
        'files': [
            'Epson_XP-352_Foto_Standard.png',
            'Epson_XP-352_Photo_High.png',
        ]
        },

        {
        'name': 'Dell - V515w',
        'files': [
            'dell_v515w_best_photo.png',
            'dell_v515w_fast_photo.png',
        ]
        },

        {
        'name': 'Ricoh - Aficio SG 3110dn',
        'files': [
            'ricoh_aficio-sg-3110dn_high_photo.png',
            'ricoh_aficio-sg-3110dn_draft_photo.png',
        ]
        },

        {
        'name': 'Lexmark - Z23',
        'files': [
            'lexmark_z23_high_normal.png',
            'lexmark_z23_best_photo.png',
        ]
        },
    ]

    # A region in the image is blacked out due to potentially biasing
    # handwritten info. we do not take samplings from this region
    blackout_mask = (2300, 0, 9700, 4600)

    file_names = []
    for pr in paper_printer_models:
        file_names += [os.path.splitext(f)[0] for f in pr['files']]

    # Gather relevant rows
    df = df[df.apply(lambda row: os.path.splitext(row['filename'].split('/')[-1])[0] in file_names, axis=1)]
    assert(len(df) == len(file_names))

    df['printer_model'] = df['printer_model'].apply(lambda v: v[0] if type(v) == list else v)
    df['filename'] = df['filename'].apply(lambda v: f'{os.path.splitext(v.split("/")[-1])[0]}.png')

    random_state = np.random.RandomState(seed=args.random_seed)

    # Get unique printer models
    printer_models = []
    for manufacturer in df['printer_manufacturer'].unique():
        models = df[df['printer_manufacturer'] == manufacturer]['printer_model'].unique()

        class_names = [f'{manufacturer}//{n}' for n in models]

        printer_models += class_names
    printer_models = np.array(list(sorted(printer_models)))
    df['class_name'] = df.apply(lambda r: f"{r['printer_manufacturer']}//{r['printer_model']}", axis=1)

    metainfo = {'printer_models': printer_models,
                    'num_classes': len(printer_models)}
    metainfo.update(vars(args))

    metainfo['num_crops_per_image'] = args.num_crops_per_image

    train_rows = []
    val_rows = []
    for printer_model_config in paper_printer_models:
        train_rows.append(df[df['filename'] == printer_model_config['files'][0]])
        val_rows.append(df[df['filename'] == printer_model_config['files'][1]])

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)

    assert(len(train_df) == len(val_df) and len(train_df) == len(file_names) / 2)

    train_df, descriptors = get_features(
        df=train_df, printer_models=printer_models,
        img_data_prefix='./images/output', num_segments=args.num_crops_per_image,
        crop_size=args.crop_size, random_state=random_state,
        feature_fn=process_segment, save_crops=True, masked_region=blackout_mask)

    random_state = np.random.RandomState(seed=args.random_seed+1)

    val_df, _ = get_features(
                df=val_df, printer_models=printer_models,
                img_data_prefix='./images/output', num_segments=args.num_crops_per_image,
                crop_size=args.crop_size, random_state=random_state,
                feature_fn=process_segment, save_crops=True, masked_region=blackout_mask)

    metainfo['num_features'] = len(descriptors)
    metainfo['feature_descriptors'] = descriptors
    metainfo['blackout_mask'] = blackout_mask

    with open(output_file, 'wb') as f:
        result = {'original_df': df, 'train_df': train_df, 'val_df': val_df, 'metainfo': metainfo}

        pickle.dump(result, f)

if __name__ == '__main__':
    run()
