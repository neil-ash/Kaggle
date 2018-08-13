"""
File to go from 1-hot arrays -> digit values
"""

import numpy as np

one_hot = np.load('submissions//K-NN//one-hot_predictions.npy')

# translate from 1-hot arrays to digits (index of 1 in array)
digit = []
unclassified = 0
for i in range(one_hot.shape[0]):
    if 1 in one_hot[i]:
        digit.append(np.where(one_hot[i] == 1)[0][0])
    else:
        digit.append(np.random.randint(0, 10))
        unclassified += 1

# convert to np array, then save to csv file
digit = np.asarray(digit)
np.savetxt('temp_predictions.csv', digit, delimiter=',', fmt='%d')

