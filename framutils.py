import glob
from shutil import copyfile
import os
import cv2
import os.path as osp
import sys
from lxml.etree import Element, SubElement, tostring
import time

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def FrameCopy(img_dir_path, out_dir):
    '''
    img_dir_path = ['/media/lih/LiHaoDIsk/Dataset/frame500','/media/lih/LiHaoDIsk/Dataset/frame800']
    out_dir = '/media/lih/LiHaoDIsk/Dataset/DatasetForTrain'
    '''
    cnt = 0
    for path in img_dir_path:
        for i in glob.glob(path+'/*.jpg'):
            frame_name = '/{:06d}.jpg'.format(cnt)
            frame_name = out_dir+frame_name
            print(frame_name)
            copyfile(i,frame_name)
            cnt +=1

def changexml(xml_path):
    # xml_path = '/home/lih/MarkerDataset/annotations'
    files = []
    for i in glob.glob(xml_path+'/*.xml'):
        files.append(i)
    print("{} files will be changed".format(len(files)))
    for xmlFile in files:
        dom = xml.dom.minidom.parse(xmlFile)
        root = dom.documentElement
        name = root.getElementsByTagName('name')
        print(xmlFile)
        for i in range(len(name)):
            print(name[i].firstChild.data)
            name[i].firstChild.data = 'marker'
        # save changed name in xmlFile
        with open(xmlFile,'w') as fh:
            dom.writexml(fh)

def dataset_init(root_path='/media/lih/LiH/FakeDataset',
                 video_path = '/media/lih/LiH/FakeDataset/videos',
                 xml_path = '/media/lih/LiH/FakeDataset/annotations',
                 img_path = '/media/lih/LiH/FakeDataset/images'):
    '''
    vis_threshold =
    '''
    video_list = []
    for i in glob.glob(video_path+'/*.mp4'):
        video_list.append(i)
        # video_name.append(os.path.basename(i))
    log_path = os.path.join(root_path,'log_4.txt')
    idx = 52300
    detector = create_detector()
    for tmp in video_list:
        cnt = 0
        VideoCapture = cv2.VideoCapture(tmp)
        _, frame = VideoCapture.read()
        start = time.time()
        while _:
            cv2.imshow('RawVideo:{}'.format(os.path.basename(tmp)),frame)
            if cnt%50==0:
                if frame.shape is not (720,1280,3):
                    frame = cv2.resize(frame,(1280,720))
                detect_result = detector.run(frame)
                ret = detect_result['results']
                total_time = detect_result['tot']
                ret = ret[1]
                bboxes = []
                frame_show = frame.copy()
                for bbox in ret:
                    if bbox[4] > 0.5:
                        # frame_show = frame.copy()
                        cv2.rectangle(frame_show,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),2)
                        # because xml start from [1,1]
                        xmin = int(bbox[0]) + 1
                        ymin = int(bbox[1]) + 1
                        xmax = int(bbox[2]) + 1
                        ymax = int(bbox[3]) + 1
                        bboxes.append([xmin, ymin, xmax, ymax])
                if len(bboxes) is not 0:
                    cv2.imshow('detect:{}'.format(os.path.basename(tmp)),frame_show)
                    img_name = "{:06d}.jpg".format(idx)
                    img_name_abs = os.path.join(img_path, img_name)
                    print("save {}".format(img_name_abs))
                    # save img
                    cv2.imwrite(img_name_abs, frame)
                    h,w,d = frame.shape
                    # print(h,w,d)
                    # create xml file
                    node_root = Element('annotation')
                    node_folder = SubElement(node_root, 'folder')
                    node_folder.text = 'images'
                    node_filename = SubElement(node_root, 'filename')
                    node_filename.text = img_name
                    node_path = SubElement(node_root, 'path')
                    node_path.text = img_name_abs
                    node_source = SubElement(node_root,'source')
                    node_database = SubElement(node_source, 'database')
                    node_database.text = 'Unknown'

                    node_size = SubElement(node_root, 'size')
                    node_width = SubElement(node_size, 'width')
                    node_width.text = str(w)

                    node_height = SubElement(node_size, 'height')
                    node_height.text = str(h)

                    node_depth = SubElement(node_size, 'depth')
                    node_depth.text = str(d)

                    node_segmented = SubElement(node_root, 'segmented')
                    node_segmented.text = '0'

                    for bbox in bboxes:
                        node_object = SubElement(node_root, 'object')
                        node_name = SubElement(node_object, 'name')
                        node_name.text = 'marker'
                        node_pose = SubElement(node_object, 'pose')
                        node_pose.text = 'Unspecified'
                        node_truncated = SubElement(node_object, 'truncated')
                        node_truncated.text = '0'
                        node_difficult = SubElement(node_object, 'difficult')
                        node_difficult.text = '0'
                        node_bndbox = SubElement(node_object, 'bndbox')
                        node_xmin = SubElement(node_bndbox, 'xmin')
                        node_xmin.text = str(bbox[0])
                        node_ymin = SubElement(node_bndbox, 'ymin')
                        node_ymin.text = str(bbox[1])
                        node_xmax = SubElement(node_bndbox, 'xmax')
                        node_xmax.text = str(bbox[2])
                        node_ymax = SubElement(node_bndbox, 'ymax')
                        node_ymax.text = str(bbox[3])

                    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
                    xml_file_path = os.path.join(xml_path,"{:06d}.xml".format(idx))
                    with open(xml_file_path,'wb') as file:
                        file.write(xml)
                    idx +=1
            _, frame = VideoCapture.read()
            cv2.waitKey(1)
            cnt += 1
        if os.path.exists(log_path):
            with open(log_path,'a+') as log:
                log.write("From {} create samples, idx = {}\n".format(os.path.basename(tmp),idx))
                print("From {} create samples, idx = {}\n".format(os.path.basename(tmp),idx))
        else:
            with open(log_path,'w+') as log:
                log.write("From {} create samples, idx = {}\n".format(os.path.basename(tmp), idx))
                print("From {} create samples, idx = {}\n".format(os.path.basename(tmp), idx))
        cv2.destroyAllWindows()


def copy_video(video_path='/media/lih/LiH/RawData', save_path='/media/lih/LiH/RenameVideos'):
    cnt = 0
    for root, dirs, files in os.walk(video_path):
        dirs.sort() # sort the dir because os.walk return arbitrary
        for name in files:
            if name[-3:] == 'mp4':
                file_path = os.path.join(root, name)
                print(file_path)
                file_name = "{:06d}.mp4".format(cnt)
                file_save_path = os.path.join(save_path, file_name)
                print(file_save_path)
                cnt +=1
                copyfile(file_path,file_save_path)
                print("Copy Done")
                with open(os.path.join(save_path,'index.txt'),'a+') as log:
                    log.write(file_path)
                    log.write('\n')
                    log.write(file_save_path)
                    log.write('\n')


def create_detector():
    this_dir = osp.dirname(__file__)
    src_path = osp.join(this_dir, 'src')
    lib_path = osp.join(src_path, 'lib')
    add_path(lib_path)
    from detectors.detector_factory import detector_factory
    from opts import opts

    opt = opts().init("--load_model {} --arch {}".format("./models/model_last_2w5.pth",
                                                         "resdcn_18").split(' '))
    detector = detector_factory[opt.task](opt)
    return detector



if __name__=='__main__':
    # dataset_init()
    # changexml('/home/lih/MarkerDataset/annotations')
    # copy_video()
    dataset_init(root_path='/media/lih/LiH/preDataset',
                 video_path='/media/lih/LiH/preDataset/videos4',
                 xml_path='/media/lih/LiH/preDataset/annotations4',
                 img_path='/media/lih/LiH/preDataset/images4')