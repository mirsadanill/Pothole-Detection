#!/usr/bin/env /home/swarmnect/pythondl/bin/python
#set python path

# ros
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

# standard
import requests
import cv2
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

# custom
from tensor_inference import tensor_engine

#--------------#
class inferenceHandler():
    def __init__(self,trtengine):
        self.bridge = CvBridge()
        self.imagecv2 = None
        self.inference = trtengine

    def uploadtos3(self): # not tested or called for now...
        url = 'http://file.api.wechat.com/cgi-bin/media/upload?access_token=ACCESS_TOKEN&type=TYPE'
        files = {'media': open(self.imagecv2, 'rb')}
        requests.post(url, files=files)

    def inferencing(self):
        try:
            self.imagecv2 = self.bridge.imgmsg_to_cv2(self.imageros, "bgr8")
            t1 = datetime.now()
            stamp = t1.strftime("%Y%m%d_%H%M_%S_%f")
            self.inference.read_frame(self.imagecv2,f"{stamp}")
            t2 = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(t2)}][INFO] camera_sub_inference.py : Inference time: {str(t2-t1)}...\n")

            self.inference.save_frame_label() # saves to dataset with label and draws bounding boxes then saves

        except CvBridgeError as e:
            print(e)
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][ERROR] camera_sub_inference.py : {e}\n")
                fp.write(f"[{str(now)}][WARNING] camera_sub_inference.py : Frame can not read. Skipping...\n")

        

    def callback(self,imageros):
        self.imageros = imageros
        self.inferencing()

    def listener(self):
        rospy.init_node('fcam_inference', anonymous=False)
        rospy.Subscriber("/fraw_img", Image, self.callback)
        try:
            rospy.spin()
        except Exception as e:
            #print("Shutting down...")
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][ERROR] camera_sub_inference.py : {e}\n")
                fp.write(f"[{str(now)}][WARNING] camera_sub_inference.py : ROS Interrupted. Retrying...\n")
        

if __name__ == '__main__':
    engine_file = "/home/swarmnect/ros/yolov4_1_tiny_3_576_768_v2.engine"
    class_names_file = "/home/swarmnect/ros/classes.txt"
    numberofclasses = 1
    image_size_h_w = [576, 768]
    batch_size = 1
    try:
        trtengine = tensor_engine(engine_file, class_names_file, "/home/swarmnect/dataset_data/", image_size_h_w=image_size_h_w, batch_size=batch_size)

    except Exception as e:
        print(e)
        now = datetime.now()
        with open(f'/home/swarmnect/ros.log', 'a') as fp:
            fp.write(f"[{str(now)}][ERROR] camera_sub_inference.py : {e}\n")
            fp.write(f"[{str(now)}][WARNING] camera_sub_inference.py : TensorRT network cannot be created from engine file: {engine_file}. Exiting...\n")
        exit(1)

    now = datetime.now()
    with open(f'/home/swarmnect/ros.log', 'a') as fp:
        fp.write(f"[{str(now)}][INFO] camera_sub_inference.py : TensorRT network (from {engine_file}) loaded.\n")
    
    while True:
        try:
            ih = inferenceHandler(trtengine)
            ih.listener()
        except rospy.ROSInterruptException:
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][WARNING] camera_sub_inference.py : ROS Interrupted. Exiting...\n")
            break
        except KeyboardInterrupt:
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][WARNING] camera_sub_inference.py : Keyboard Interruption... Exiting...\n")
            break
        except Exception as e:
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][ERROR] camera_sub_inference.py : {e}\n")
                fp.write(f"[{str(now)}][WARNING] camera_sub_inference.py : Retrying...\n")
            continue