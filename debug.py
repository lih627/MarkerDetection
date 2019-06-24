import pathlib
import glob
import cv2
import shutil
img_path = '/media/lih/LiHaoDIsk/DatasetForTrain/Dataset/'
file = []

for i in glob.glob('/media/lih/LiHaoDIsk/DatasetForTrain/Dataset/*.jpg'):
    file.append(i)

target = '/home/lih/Mydataset'

length = len(file)
cnt = 0

for tmp in file:
    if cnt%300 is 0 :
        print("copy:{}".format(cnt))
        shutil.copy(tmp,target)
    cnt += 1