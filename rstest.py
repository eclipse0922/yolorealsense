

import time

from pydarknet import Detector, Image

import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2



config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()

depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)


if __name__ == "__main__":

    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        start_time = time.time()

        color_image = np.asanyarray(color_frame.get_data())
        pydark_image=Image(color_image)
        results = net.detect(pydark_image)

        del pydark_image
        for cat, score, bounds in results:

           x, y, w, h = bounds
           depthVal = depth_frame.get_distance(int(x),int(y))
           depthVal =round(depthVal,2)
           cv2.rectangle(color_image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,127),2)
           cv2.putText(color_image, str(cat.decode("utf-8")+" in "+str(depthVal)+"(meter)"), (int(x-w/2 ), int(y-h/2)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 127))

        cv2.namedWindow("preview", cv2.WINDOW_NORMAL);
        cv2.resizeWindow("preview", 1920, 1080)
        cv2.imshow("preview", color_image)

        end_time = time.time()

        print("Elapsed Time:", end_time - start_time)

        key = cv2.waitKey(1)

        if key == 0xFF & ord("q"):
            pipeline.stop()
            break
