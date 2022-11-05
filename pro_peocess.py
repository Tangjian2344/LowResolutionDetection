import os
import cv2
import numpy as np

pre = '/home/tangjian/Codes/VOC/images/'
datas = ['test2007', 'train2007', 'train2012']


# check the images size
# max_h, max_w = 0, 0
# for data in datas:
#     path = pre + data
#     files = os.listdir(path)
#     os.chdir(path)
#     for file in files:
#         if not file.endswith('pkl'):
#             img = cv2.imread(file)
#             h, w = img.shape[:2]
#             if h > max_h: max_h = h
#             if w > max_w: max_w = w
#             # if h > 512 or w > 512:
#             #     print('the {} in {} size is so big'.format(file, path))
#             # else:
#             #     print('size is normal')
# print(max_h, max_w)

for data in datas:
    path = pre + data
    files = os.listdir(path)
    os.chdir(path)
    for file in files:
        if not file.endswith('pkl'):
            img = cv2.imread(file)
            h, w = img.shape[:2]
            if h > 512 or w > 512:
                print('the {} in {} size is so big'.format(file, path))
                break
            else:
                # print('size is normal')
                ph = 512 - h
                pw = 512 - w
                img = np.pad(img, ((0, ph), (0, pw), (0, 0)),'mean')
                d_path = path.replace('images', 'images512_') + '/' + file
                cv2.imwrite(d_path, img)
                pass