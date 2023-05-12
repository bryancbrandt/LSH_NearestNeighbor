# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

import numpy as np
import pickle

# Create 10 sets of uniform random distributions ranging in value from 0 to 255 across 1024 dimensions
# rng = np.random.RandomState(0)
# X = rng.uniform(0, 255, (10, 1024))


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


file_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

for names in file_list:
    file1 = unpickle(names)
    filenames = file1[b'filenames']
    red_data = file1[b'data'][:, :1024]
    green_data = file1[b'data'][:, 1024:2048]
    blue_data = file1[b'data'][:, 2048:]

    with open("/image_data/blue_data.csv", "a") as redfile:
        row_count = 0
        for rows in blue_data:
            # redfile.write(filenames[row_count].decode('ascii'))
            # redfile.write(",")
            list = rows.astype(str)
            for items in list:
                redfile.write(items)
                redfile.write(",")
            redfile.write("\n")
            row_count = row_count + 1

