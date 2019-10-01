import os
import sys
import pickle
import argparse

from numpy.lib.format import open_memmap
from hmdata import TotalCapture
from torch.utils.data import DataLoader

max_body = 1
num_joint = 18
max_frame = 300
pace = 50


def action2label(action):
    action = action[:-1]
    if action == 'acting':
        return 0
    elif action == 'freestyle':
        return 1
    elif action == 'rom':
        return 2
    elif action == 'walking':
        return 3

def gendata(data, out_path, part):
    for 

if __name__ == '__main__':

    import ptvsd
    ptvsd.enable_attach(address=('localhost'))
    ptvsd.wait_for_attach()

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/mnt/md0/yinw/project/data/totalcapture')
    parser.add_argument('--out_folder', default='data/TotalCapture')
    arg = parser.parse_args()

    tp_train_data = TotalCapture(arg.data_path, cues='imu', mode='debug')