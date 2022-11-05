import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

src = '/home/tangjian/Codes/plots/'

# 可视化1
# Method = ['Truth', 'BaseDHR', 'BaseDLR', 'Ours_SR']
#
# names = os.listdir(src + Method[0])
# names.sort()
#
# for name in names:
#     for i in range(len(Method)):
#         if i == 0:
#             img = cv2.imread(src + Method[i] + '/' + name)
#         else:
#             t = cv2.imread(src + Method[i] + '/' + name)
#             # print(img.shape, t.shape)
#             img = np.hstack((img, t))
#
#     cv2.imwrite(src + 'visual/' + name, img)

# 可视化2

# Method = ['Truth', 'BlurHR', 'BlurLR', 'BlurSR']
#
# names = os.listdir(src + Method[0])
# names.sort()
#
# for name in names:
#     for i in range(len(Method)):
#         if i == 0:
#             img = cv2.imread(src + Method[i] + '/' + name)
#         else:
#             img = np.hstack((img, cv2.imread(src + Method[i] + '/' + name)))
#
#     cv2.imwrite(src + 'visual_blur/' + name, img)

# 可视化 3

# Method = ['Truth', 'NoiseHR', 'NoiseLR', 'NoiseSR']
#
# names = os.listdir(src + Method[0])
# names.sort()
#
# for name in names:
#     for i in range(len(Method)):
#         if i == 0:
#             img = cv2.imread(src + Method[i] + '/' + name)
#         else:
#             img = np.hstack((img, cv2.imread(src + Method[i] + '/' + name)))
#
#     cv2.imwrite(src + 'visual_noise/' + name, img)

