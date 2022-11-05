import os
import pickle
import imageio
import cv2

# paths = [
#          '/home/tangjian/Codes/VOC/images512/test2007',
#          '/home/tangjian/Codes/VOC/images512/train2012',
#          '/home/tangjian/Codes/VOC/images512/train2007',
#          '/home/tangjian/Codes/VOC/images512/test2007_lr',
#          '/home/tangjian/Codes/VOC/images512/train2012_lr',
#          '/home/tangjian/Codes/VOC/images512/train2007_lr',
# ]

paths = ['/home/tangjian/Codes/VOC/images512/test2007_BISR/']

for path in paths:
    files = os.listdir(path)
    os.chdir(path)
    for file in files:
        bin_file = file.split('.')[0] + '.pkl'
        if not os.path.exists(bin_file):
            print('make pkl file: {}: {}'.format(path, bin_file))
            with open(bin_file, 'wb') as f:
                pickle.dump(cv2.imread(file), f)
            pass


# for path in paths:
#     files = os.listdir(path)
#     os.chdir(path)
#     for file in files:
#         if file.endswith('pkl'):
#             os.remove(file)

# a = "/home/tangjian/Codes/VOC/images/test2012/2008_000001.jpg"
# b = "/home/tangjian/Codes/VOC/images/test2012/2008_000001.pkl"
#
# a = imageio.imread(a)
# with open(b, 'rb') as f:
#     b = pickle.load(f)
# print((a == b).sum())
