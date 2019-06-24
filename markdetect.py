import sys
import os.path as osp
import glob
import cv2



def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)



def markerdetector(detector, img, vis_threshold=0.3,show_img = True):
    this_dir = osp.dirname(__file__)
    src_path = osp.join(this_dir,'src')
    lib_path = osp.join(src_path,'lib')
    add_path(lib_path)
    #
    # from detectors.detector_factory import detector_factory
    # from opts import opts
    #
    # opt = opts().init("--load_model {} --arch {}".format("./models/model_last.pth",
    #                                                      "resdcn_18").split(' '))
    # detector = detector_factory[opt.task](opt)
    ret = detector.run(img)['results']
    ret = ret[1]
    result = []
    for bbox in ret:
        if bbox[4] > vis_threshold:
            if show_img:
                cv2.rectangle(img,(bbox[0],bbox[1]),
                              (bbox[2],bbox[3]),(255,0,0),2)
                x_center = (bbox[0]+bbox[2])/2
                y_center = (bbox[1]+bbox[3])/2
                cv2.circle(img,(int(x_center),int(y_center)),1,(255,0,0),8)
                result.append([x_center, y_center])
    cv2.imshow('cdet',img)
    cv2.waitKey(0)
    return img, result






if __name__=='__main__':

    this_dir = osp.dirname(__file__)
    src_path = osp.join(this_dir, 'src')
    lib_path = osp.join(src_path, 'lib')
    add_path(lib_path)
    from detectors.detector_factory import detector_factory
    from opts import opts

    opt = opts().init("--load_model {} --arch {}".format("./models/model_last.pth",
                                                         "resdcn_18").split(' '))
    detector = detector_factory[opt.task](opt)

    img_path = '/home/lih/MarkerDataset/images'
    file = []
    for i in glob.glob(img_path+'/*.jpg'):
        file.append(i)
    print(len(file))
    for tmp in file:
        img = cv2.imread(tmp)
        # cv2.imshow('img',img)
        markerdetector(detector, img, vis_threshold=0.3)