import json 

with open('/home/yuxinyao/body_models/Mosh_related/smplx_markerset_new.json') as f:
        marker_grab_67 = json.load(f)['markersets']

# indices = []
# required_type = ['body','hand','foot','finger']

# for i,d in enumerate(marker_grab_67):
#     if d['type'] in required_type:
#         indices = indices + list(d['indices'].values())

# print(indices)

def get_grab_marker(required_type):
    with open('/home/yuxinyao/body_models/Mosh_related/smplx_markerset_new.json') as f:
        marker_grab_67 = json.load(f)['markersets']
    indices = []
    for i,d in enumerate(marker_grab_67):
        if d['type'] in required_type:
            indices = indices + list(d['indices'].values())
    return indices
