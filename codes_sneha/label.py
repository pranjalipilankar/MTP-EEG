import numpy as np

data = np.load("/home/ab_students/EEG-MTP/DATA/SEED/arr_0.npy", allow_pickle=True)

print(data[-200:])
print(type(data))
print(data.shape)
print(data.dtype)
