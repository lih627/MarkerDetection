import sys
import os.path as osp
import glob
import cv2



def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)



def markerdetector(detector, img, vis_threshold=0.3,show_img = True):
    img = cv2.resize(img, (640, 360))
    output = detector.run(img)
    ret = output['results']
    print("tot time {}".format(output['tot']))
    ret = ret[1]
    result = []
    for bbox in ret:
        if bbox[4] > vis_threshold:
            if show_img:
                cv2.rectangle(img,(bbox[0],bbox[1]),
                              (bbox[2],bbox[3]),(255,0,0),2)
                x_center = (bbox[0]+bbox[2])/2
                y_center = (bbox[1]+bbox[3])/2
                cv2.circle(img,(int(x_center),int(y_center)),1,(255,0,0),1)
                result.append([x_center, y_center])
    img = cv2.resize(img,(1280,640))
    cv2.imshow('cdet',img)
    cv2.waitKey(2)
    return img, result






if __name__=='__main__':

    this_dir = osp.dirname(__file__)
    src_path = osp.join(this_dir, 'src')
    lib_path = osp.join(src_path, 'lib')
    add_path(lib_path)
    from detectors.detector_factory import detector_factory
    from opts import opts

    opt = opts().init("--load_model {} --arch {}".format("./models/model_last_lr_2w5.pth",
                                                         "resdcn_18").split(' '))
    detector = detector_factory[opt.task](opt)

    img_path = '/home/lih/MarkerDataset/images'
    file = []
    # for i in glob.glob(img_path+'/*.jpg'):
    #     file.append(i)
    # print(len(file))
    # for tmp in file:
    #     img = cv2.imread(tmp)
    #     # cv2.imshow('img',img)
    #     # img = cv2.resize(img,(640, 320))
    #     markerdetector(detector, img, vis_threshold=0.5)
    print('begin detect')
    video_path = '/home/lih/project/MarkerDetection/images/test_night.mp4'
    cam = cv2.VideoCapture(video_path)
    _, frame = cam.read()
    cnt = 0
    while _:
        cv2.imshow('original', frame)
        print(frame.shape)
        if cnt%1 ==0:
            markerdetector(detector, frame, vis_threshold=0.5)
        _, frame = cam.read()
        cnt+=1
    cv2.destroyAllWindows()