import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM

import cv2

import threading
import termios
import contextlib
import sys

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='/Users/alekseikurnosov/Documents/GitHub/EMO-AffectNetModel/video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--instant_result', type=str, default='report/emo.emo', help='The file to write minireports')
parser.add_argument('--path_FE_model', type=str, default='models/EmoAffectnet/weights_0_66_37_wo_gl.h5',
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default='models/LSTM/SAVEE_with_config.h5',
                    help='Path to a model for emotion prediction')

args = parser.parse_args()

needStop = False

label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

@contextlib.contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

def key_listen():
    global needStop
    print('Stop with Q')
    with raw_mode(sys.stdin):
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch or ch == 'q' or ch == 'Q':
                    break
        except (KeyboardInterrupt, EOFError):
            pass
        
    needStop = True

def pred_one_video(path):
    start_time = time.time()
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, total_frame = detect.get_frames()

    if not os.path.exists(args.path_save):
        os.makedirs(args.path_save)
        
    filename = os.path.basename(path)[:-4] + '.csv'
    
    df, mode = predict(dict_face_areas, total_frame)

    df.to_csv(os.path.join(args.path_save,filename), index=False)
    end_time = time.time() - start_time
    
    print('Report saved in: ', os.path.join(args.path_save,filename))
    print('Predicted emotion: ', label_model[mode])
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()

def predict(dict_face_areas, total_frame):
    name_frames = list(dict_face_areas.keys())
    face_areas = list(dict_face_areas.values())
    #print("----------------")
    #print("name_frames: ", name_frames)
    #print("----------------")
    #print("face_areas in stack: ", np.stack(face_areas))
    #print("----------------")
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    features = EE_model(np.stack(face_areas))
    seq_paths, seq_features = sequences.sequences(name_frames, features)
    pred = LSTM_model(np.stack(seq_features)).numpy()
    #print("pred: ", pred)
    #print("----------------")

    all_pred = []
    all_path = []
    for id, c_p in enumerate(seq_paths):
        c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1])+1)]
        c_pr = [pred[id]]*len(c_f)
        all_pred.extend(c_pr)
        all_path.extend(c_f)    
    m_f = [str(i).zfill(6) for i in range(int(all_path[-1])+1, total_frame+1)] 
    m_p = [all_pred[-1]]*len(m_f)
    
    df=pd.DataFrame(data=all_pred+m_p, columns=label_model)

    #print("----------------")
    #print("df from pd.DataFrame: ", df)
    df['frame'] = all_path+m_f
    df = df[['frame']+ label_model] 
    #print("----------------")
    #print("df before grouping: ", df)
    df = sequences.df_group(df, label_model)
    #print("----------------")
    #print("df: ", df)
    #print("----------------")

    mode = stats.mode(np.argmax(pred, axis=1))[0]
    return (df, mode)

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in enumerate(path_all_videos):
        if "video" in cr_path:
            print('{}/{}'.format(id+1, len(path_all_videos)))
            pred_one_video(os.path.join(args.path_video,cr_path))
        
def start_pred_camera():
    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    save_video_path = os.path.join(args.path_video, "output.mp4")
    #print("In params save_video_path: ", save_video_path)
    conf_int = 1
    detect = get_face_areas.VideoCamera(conf=0.7)
    detect.init_camera()
    needStop = False

    mode = 0
    prev_mode = -1
    
    while not needStop:
        current_frame = detect.get_current_camera_frame()
        if len(current_frame) > 0:
            df, mode = predict(current_frame, 1)
            predicted_weights = df.values.tolist()[0][1: ]
            if predicted_weights[mode] < conf_int * 0.1:
                mode = 0
            predicted = list(map(lambda probability: round(probability, 2), predicted_weights))

            print("mood: ", label_model[mode], "predict: ", predicted)
        else:
            mode = 0

        if mode != prev_mode:
            prev_mode = mode
            save_minireport(mode)

        key = cv2.waitKey(20)
        match key:
            case 61:
                if conf_int < 9:
                    conf_int += 1
                    print(">>>>conf is set to, ", 0.1 * conf_int)
            case 45:
                if conf_int > 1:
                    conf_int -= 1
                    print(">>>>conf is set to, ", 0.1 * conf_int)
            case 27:
                needStop = True
    
    detect.save_result()

def save_minireport(mode):
    print("emo_path:", args.instant_result)
    f = open(args.instant_result, "w")
    f.write(str(mode))
    f.close()
        
if __name__ == "__main__":
    #pred_all_video()
    start_pred_camera()
    #key_listen()

    #t1 = threading.Thread(target=key_listen)
    #t2 = threading.Thread(target=start_pred_camera)

    #t1.start()
    #t2.start()

    #t1.join()
    #t2.join()