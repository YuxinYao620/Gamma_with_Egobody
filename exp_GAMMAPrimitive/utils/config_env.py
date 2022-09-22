import yaml
import os, sys
import socket

def get_host_name():
    return hostname


def get_body_model_path():
    print(hostname)
    # if 'vlg-atlas' in hostname:
    #     bmpath = '/home/yuxinyao/body_models/'
    # elif 'emerald' in hostname:
    #     bmpath = '/home/yuxinyao/body_models/'
    # else:
    #     raise ValueError('not stored here')
    bmpath = '/home/yuxinyao/body_models/'
    return bmpath

def get_body_marker_path():
    # if 'vlg-atlas' in hostname:
    #     mkpath = '/home/yuxinyao/body_models/Mosh_related'
    # elif 'emerald' in hostname:
    #     mkpath = '/home/yuxinyao/body_models/Mosh_related'
    # else:
    #     raise ValueError('not stored here')
    mkpath = '/home/yuxinyao/body_models/Mosh_related'
    return mkpath

def get_amass_canonicalized_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-data/AMASS-Canonicalized-MP/data'
    elif 'emerald' in hostname:
        mkpath = '/mnt/hdd/datasets/AMASS_SMPLH_G-canon/data'
    elif 'yuxinyao' in hostname:
        # mkpath = '/home/yuxinyao/datasets/egobody/canicalized-camera-wearer'
        mkpath = '/home/yuxinyao/datasets/egobody/'
    else:
        raise ValueError('not stored here')
    return mkpath

def get_amass_canonicalizedx10_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-data/AMASS-Canonicalized-MPx10/data'
    elif 'emerald' in hostname:
        mkpath = '/home/yzhang/Videos/AMASS-Canonicalized-MPx10/data'
    elif 'yuxinyao' in hostname:
        # mkpath = '/home/yuxinyao/datasets/egobody/canicalized-camera-wearer_x3'
        mkpath = '/home/yuxinyao/datasets/egobody/'
    else:
        raise ValueError('not stored here')
    return mkpath




hostname = socket.gethostname()



















