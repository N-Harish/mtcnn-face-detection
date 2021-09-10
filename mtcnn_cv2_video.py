import cv2
from mtcnn_cv2 import MTCNN
import argparse
import warnings
import time
import argparse


ag = argparse.ArgumentParser()
ag.add_argument("-v", "--video", required=True, default='../data/video_input/face.mp4', help="Enter the path of test video")
ag.add_argument("-o", "--output", required=True, default='../data/video_output/mtcnn_face_fps.mp4', help="Enter the path of output video")

ap = vars(ag.parse_args())


warnings.filterwarnings('ignore')


detector = MTCNN()
cap = cv2.VideoCapture(ap['video'])
cnt = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

results = cv2.VideoWriter(ap['output'],
                           cv2.VideoWriter_fourcc(*'XVID'),
                           10,
                           size)

prev_frame_t = 0
new_frame_t = 0

while cap.isOpened():
    ret, frame = cap.read()
    # check ret to see if frame is there
    if ret:
        result = detector.detect_faces(frame)
        # check if model detects face
        if result:
            for res in result:
                bbox = res['box']
                conf = float(res['confidence'])
                text = "{:.2f}%".format(conf * 100)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 155, 255), 2)
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            print(f"detected {cnt}")
            cnt += 1

        # calculate fps
        new_frame_t = time.time()
        fps = 1/(new_frame_t - prev_frame_t)
        prev_frame_t = new_frame_t
        fps = str(int(fps))
        fps = "FPS:- " + fps
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 255), 3, cv2.LINE_AA)

        # write video
        results.write(frame)

        # show detected frame
        cv2.imshow('frame', frame)

        # quit key
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    else:
        break


cap.release()
results.release()
cv2.destroyAllWindows()
