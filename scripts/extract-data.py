from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.image as img
import json
import numpy as np
from tqdm import tqdm
import h5py

from multiprocessing import Pool
import time
from concurrent.futures import ThreadPoolExecutor
import functools
import pandas as pd

# Helper Functions from EDA
def ejson(p, fn): # extract json
    with open((p/fn).as_posix()) as f: return json.load(f) 
## Simple Json Reading Functions
def info_data(p): return ejson(p, 'info.json')
def dot_data(p): return ejson(p, 'dotInfo.json')
def frame_data(p): return ejson(p, 'frames.json')    
def screen_data(p): return ejson(p, 'screen.json')

def get_eye_info(i): return ejson(i, 'appleLeftEye.json'), ejson(i, 'appleRightEye.json')
def get_face_info(i): return ejson(i, 'appleFace.json')
def get_facegrid(i): return ejson(i, 'faceGrid.json')

def get_frame(p, img_fn): return img.imread(p/'..'/'..'/'gazecapture-224x224'/p.name/'frames'/img_fn)
## Larger Helper Functions
def coordinate_data(p): 
    data = dot_data(p)
    return data['XCam'], data['YCam'] # we want relative to camera coords
def orientation_data(p):
    sdata = screen_data(p)
    return sdata['Orientation']

dset_path = Path('../../gazecapture/')

# - home button spot (right / left)
# right = 3
# left = 4
file_names = "landscape-r"
O = [3]

# # Extract File Names For Keras Generator
test_size = 0.1

def extract_data(cases, accepted_o=[1, 2, 3, 4]):
    fnames = []
    XCam, YCam = [], []
    FaceH, FaceW, FaceX, FaceY = [], [], [], []
    IsValid = []
    for case in cases:
        FRAME_N = frame_data(case)
        O = orientation_data(case)
        XCAM, YCAM = coordinate_data(case)
        FACE = get_face_info(case)

        for frame_n, o, xcam, ycam, \
            fh, fw, fx, fy, valid in zip(FRAME_N, O, XCAM, YCAM, \
                                 FACE['H'], FACE['W'], FACE['X'], FACE['Y'], FACE['IsValid']):
            if o in accepted_o:
                fnames.append('{}/frames/{}'.format(case.name, frame_n))
                XCam.append(xcam)
                YCam.append(ycam)
                IsValid.append(valid)
                FaceH.append(fh)
                FaceW.append(fw)
                FaceX.append(fx)
                FaceY.append(fy)

    # package to dataframe
    df = pd.DataFrame(data={'file_names': fnames, 
                            'XCam': XCam,
                            'YCam': YCam,
                            'IsValid': IsValid,
                            'FaceH': FaceH, 
                            'FaceW': FaceW,
                            'FaceX': FaceX,
                            'FaceY': FaceY})        
    
    return df


# Extract all case names first to split => faces it hasnt seen before
train_cases, test_cases = train_test_split(list(dset_path.iterdir()), test_size=test_size)

train_df = extract_data(train_cases, accepted_o=O)
test_df = extract_data(test_cases, accepted_o=O)
train_df.to_csv('{}-traindf.csv'.format(file_names))
test_df.to_csv('{}-testdf.csv'.format(file_names))
 