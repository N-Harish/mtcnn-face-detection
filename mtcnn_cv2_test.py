import cv2
from mtcnn_cv2 import MTCNN
import argparse
import warnings

warnings.filterwarnings('ignore')


arg = argparse.ArgumentParser()
arg.add_argument("-c", "--cam", required=True, type=int, help='Please input your camera no')
args = vars(arg.parse_args())

cam = args['cam']

detector = MTCNN()
cap = cv2.VideoCapture(cam)
cnt = 0


while cap.isOpened():
    ret, frame = cap.read()
    result = detector.detect_faces(frame)
    if result:
        for res in result:
            bbox = res['box']
            conf = float(res['confidence'])
            text = "{:.2f}%".format(conf * 100)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 155, 255), 2)
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        print("detected")
        cv2.imwrite(f'frame{cnt}.jpg', frame)
        cnt += 1
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
