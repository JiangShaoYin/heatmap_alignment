#*coding=utf-8
from __future__ import print_function
import os
import cv2
import numpy as np
from collections import deque

SUPPORTED_VIDEO_TYPES = ['ts', 'mkv', 'mp4', 'm2ts', 'flv']
SUPPORTED_IMAGE_TYPES = ['jpg', 'png', 'jpeg', 'bmp']

class VideoLoader(object):
    """MyVideo 类，封装过的openCv读取到的视频及一些常用操作
    """

    def __init__(self, sourceVideoFileName, bufferSize=1):
        self.sourceVideoFileName = sourceVideoFileName
        self.bufferSize = bufferSize
        self.captureVideo()
        self.initVideoInfo()

    def initVideoInfo(self):
        """读取视频基本信息
        """
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        
        
    def captureVideo(self):
        """封装的获取视频句柄"""
        self.cap = cv2.VideoCapture(self.sourceVideoFileName)
        self.currentFrameIndex = 0
        self.frameBuffer = deque(maxlen=self.bufferSize)
    
    def reCapVideo(self):
        """重新获取视频句柄
        """
        self.cap.release()
        self.captureVideo()
        self.initVideoInfo()
        self.initRectsInfo()

    def read(self):
        """读取当前视频图像
        """
        print("Load Frame: %d"%self.currentFrameIndex)
        self.currentFrameIndex += 1
        succ, frame = self.cap.read()
        if not succ:
            return None, None, None

        def split_yuv(img):
            yuv = cv2.cvtColor(frame,
                        cv2.COLOR_BGR2YUV_I420)
            yuv = yuv.flatten()
            w = self.frameWidth
            h = self.frameHeight

            Y = yuv[:w*h].reshape(h, w)
            U = yuv[w*h:w*h*5/4].reshape(h/2, w/2)
            V = yuv[w*h*5/4:].reshape(h/2, w/2)
            return Y, U, V
 
        y, u, v = split_yuv(frame)
        self.frameBuffer.append(y.copy())

        return (y, u, v)

    def getSize(self):
        """获取视频大小
        """
        return (self.frameWidth, self.frameHeight)

    def close(self):
        """释放视频句柄
        """
        self.cap.release()
        self.currentFrameIndex = 0
    
    def getAverageFrame(self):
        """计算平均帧
        """
        w = self.frameWidth
        h = self.frameHeight
        averageFrame = np.zeros([h, w], dtype=np.float) 

        for frame in self.frameBuffer:
            averageFrame += frame.astype(np.float)

        averageFrame /= len(self.frameBuffer)
        return averageFrame.astype(np.uint8)

class VideoWriter:
    def __init__(self, outputFile):
        self.yuvwriter = open(outputFile, "wb")

    def writergb(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420) 
        self.yuvwriter.write(yuv.data)
    
    def writeyuv(self, y, u, v):
        h, w = y.shape[:2]

        yuv = np.concatenate([
                    y.flatten(),
                    u.flatten(),
                    v.flatten()])
        #yuv = np.vstack([y, np.hstack([u, v])])
        self.yuvwriter.write(yuv.data)
        

    def close(self):
        self.yuvwriter.close()
    

######################################################
def gen_filename():
    filename = str(uuid1())+".jpg"
    return filename

def valid_video_type(filename):
    filetype = filename.split('.')[-1].lower()
    return filetype in SUPPORTED_VIDEO_TYPES 

def get_video_list(indir):
    video_files = os.listdir(indir)
    
    video_files = [f for f in video_files if valid_video_type(f)]
    video_files = [os.path.join(indir, f) for f in video_files]
    return video_files

def get_img_list(indir, types=SUPPORTED_IMAGE_TYPES):
    imgpath_list = []
    for maindir, subdirs, filenames in os.walk(indir):
        for filename in filenames:
            filetype = filename.split('.')[-1].lower()
            if filetype not in types:
                continue
            filepath = os.path.join(maindir, filename)
            imgpath_list.append(filepath)
    return imgpath_list

def video_engine(inpath, interval=1):
    if os.path.isdir(inpath):
        video_list = get_video_list(inpath)
    elif valid_video_type(inpath):
        video_list = [inpath]
    else:
        raise TypeError

    for video_path in video_list:
        cap = cv2.VideoCapture(video_path)
        idx = -1
        while True:
            idx+=1
            succ, frame = cap.read()
            if not succ:
                break
            
            if idx%interval != 0:
                continue
            yield frame

class image_engine:
    def __init__(self, indir):
        self.img_list = get_img_list(indir)

    def __len__(self):
        return len(self.img_list)

    def __iter__(self):
        return self.next()

    def next(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path)
            if img is None:
                continue
            yield img_path, img


class image_pack_engine:
    def __init__(self):
        self.datapack = None
    
    @staticmethod
    def save(imgdir, output_filename, quality=99):
        import pickle
        from tqdm import tqdm

        engine = image_engine(imgdir)
        imglist = []
        sum_size = 0
        imgiter = tqdm(engine)

        def calc_buffer_size(bytenum):
            g_num = bytenum>>30
            m_num = (bytenum-(g_num<<30))>>20
            k_num = (bytenum-(bytenum>>20<<20))>>10
            s = '%4dG,%4dM,%4dK'%(g_num, m_num, k_num)
            return s
        for i, (imgpath, img) in enumerate(imgiter):
            imgdata = cv2.imencode(
                        '.jpg',
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
            imglist.append(imgdata)
            sum_size += len(imgdata)
            imgiter.set_description('Size: {}'.format(calc_buffer_size(sum_size))) 

        with open(output_filename, "w") as f:
            pickle.dump(imglist, f, -1)

    def load(self, pkl_filename):
        import pickle
        with open(pkl_filename, "r") as f:
            self.datapack = pickle.load(f)

    def __len__(self):
        return len(self.datapack)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        imgdata = self.datapack[i]
        return cv2.imdecode(np.array(imgdata), cv2.IMREAD_UNCHANGED) 


