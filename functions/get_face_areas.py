import cv2
import numpy as np
import os
from functions import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import img_to_array
from batch_face import RetinaFace

class VideoCamera(object):
    def __init__(self, path_video='', save_video_path='', conf=0.7):
        self.path_video = path_video
        self.save_video_path = save_video_path
        if save_video_path != "":
            print("Video writer will write to: ", save_video_path)
            self.video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        self.conf = conf
        self.cur_frame = 0
        self.video = None
        self.dict_face_area = {}
        self.detector = RetinaFace()

    def __del__(self):
        if self.save_video_path != "":
            print("Trying to save video (del)...")
            self.video_writer.release()
        self.video.release()
        cv2.destroyAllWindows()

    def save_result(self):
        if self.save_video_path != "":
            print("Trying to save video...")
            self.video_writer.release()
        
    def preprocess_image(self, cur_fr):
        cur_fr = utils.preprocess_input(cur_fr, version=2)
        return cur_fr
        
    def channel_frame_normalization(self, cur_fr):
        cur_fr = cv2.cvtColor(cur_fr, cv2.COLOR_BGR2RGB)
        cur_fr = cv2.resize(cur_fr, (224,224), interpolation=cv2.INTER_AREA)
        cur_fr = img_to_array(cur_fr)
        cur_fr = self.preprocess_image(cur_fr)
        return cur_fr
            
    def get_frames(self):
        print()
        print(self.path_video)
        self.video = cv2.VideoCapture(self.path_video)
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        need_frames = list(range(1, 100+1, 10))
        print('Name video: ', os.path.basename(self.path_video))
        print('Number total of frames: ', total_frame)
        print('FPS: ', fps)
        print('Video duration: {} s'.format(np.round(total_frame/fps, 2)))
        print('Frame width:', w)
        print('Frame height:', h)
        while True:
            _, self.fr = self.video.read()
            if self.fr is None: break
            self.cur_frame += 1
            name_img = str(self.cur_frame).zfill(6)
            
            if self.cur_frame % 6 == 1:
                faces = self.detector(self.fr, cv=False)
                for f_id, box in enumerate(faces):
                    box, _, prob = box
                    if prob > self.conf:
                        startX = int(box[0])
                        startY = int(box[1])
                        endX = int(box[2])
                        endY = int(box[3])
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                        cur_fr = self.fr[startY: endY, startX: endX]
                        self.dict_face_area[name_img] = self.channel_frame_normalization(cur_fr)
                        #print("---current: ", self.dict_face_area[name_img])
        del self.detector          
        return self.dict_face_area, total_frame
    
    def init_camera(self):
        self.video = cv2.VideoCapture(0)
    
    def get_current_camera_frame(self):
        dict_face_area = {}

        if not self.video.isOpened():
            print("Camera is closed!")
            return dict_face_area
        #print('Recognizing emotion from frame...')
        current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        offset_ms = self.video.get(cv2.CAP_PROP_POS_MSEC)
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #print('Current frame: ', current_frame)
        #print('FPS: ', fps)
        #print('Current video duration: ', offset_ms)
        #print('Frame width:', w)
        #print('Frame height:', h)

        success, fr = self.video.read()
        if not success or fr is None:
            print("Read frame error...")
            return dict_face_area
        
        if self.save_video_path != "":
            self.video_writer.write(fr)

        cv2.imshow('frame', fr)
        
        faces = self.detector(fr, cv=False)

        if len(faces) < 1:
            return dict_face_area
        
        box, _, prob = faces[0]

        if prob > self.conf:
            startX = int(box[0])
            startY = int(box[1])
            endX = int(box[2])
            endY = int(box[3])
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            cur_fr = fr[startY: endY, startX: endX]
            
            name_img = str(1).zfill(6)
            dict_face_area[name_img] = self.channel_frame_normalization(cur_fr)
        
        return dict_face_area