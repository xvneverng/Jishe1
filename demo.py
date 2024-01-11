import time
from AIDetector_pytorch import Detector
import imutils
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch



def main():

    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture('D:\yolov8/012.mp4')

    fps = int(cap.get(5))
    print('fps:', fps)
    t = 8


    videoWriter = None

    while True:
        start_time = time.time()
        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im)
        result = result['frame']
        if result is not None:
            scale_factor = 1  # 缩放因子，您可以根据需要调整此值
            resized_frame = cv2.resize(result, None, fx=scale_factor, fy=scale_factor)
            #result = imutils.resize(result, height=500)
        else:
            print("Error: Image could not be loaded.")
        elapsed_time = time.time() - start_time
        frame_fps = 1 / elapsed_time
        #cv2.putText(resized_frame, 'FPS: {:.2f}'.format(frame_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        #cv2.resize(result,(2560,1080))
        
        cv2.imshow(name, resized_frame)

        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()