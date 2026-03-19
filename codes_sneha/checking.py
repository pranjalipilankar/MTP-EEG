import numpy as np

path = "/home/ab_students/EEG-MTP/DATA/SEED/All_video_label.npy"
data = np.load(path, allow_pickle=True)

print(type(data))
print(data.shape)
print(data)
