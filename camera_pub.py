#!/usr/bin/env /home/swarmnect/pythondl/bin/python
#set python path

# ros
from datetime import datetime
import time
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

# standard
import cv2
from cv_bridge import CvBridge, CvBridgeError

# custom



#--------------#
# camera params

def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60,
                       flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
                capture_width, capture_height, framerate, flip_method, display_width, display_height))

def camera_pub(imgcv2):
    skip =False
    try:
        img = bridge.cv2_to_imgmsg(imgcv2, "bgr8")
    except CvBridgeError as e:
        skip = True
        print(e)
        now = datetime.now()
        with open(f'/home/swarmnect/ros.log', 'a') as fp:
            fp.write(f"[{str(now)}][ERROR] camera_pub.py : {e}\n")
            fp.write(f"[{str(now)}][WARNING] camera_pub.py : CvBridgeError! This capture is skipping...\n")

    if not skip:
        if not rospy.is_shutdown():
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][INFO] camera_pub.py : Image published.\n")

            rospy.loginfo("Image published")
            pub.publish(img)
            rate.sleep()

def main():
    now = datetime.now()
    with open(f'/home/swarmnect/ros.log', 'a') as fp:
        fp.write(f"[{str(now)}][INFO] camera_pub.py : Node created.\n")

    while True:
        cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                rospy.logwarn("Camera read failed!")
                now = datetime.now()
                with open(f'/home/swarmnect/ros.log', 'a') as fp:
                    fp.write(f"[{str(now)}][ERROR] camera_pub.py : Camera cannot read. Camera pipeline restarting...\n")
                time.sleep(1)
                break
            try:
                camera_pub(frame)
            except rospy.ROSInterruptException:
                return
            except Exception as e:
                now = datetime.now()
                with open(f'/home/swarmnect/ros.log', 'a') as fp:
                    fp.write(f"[{str(now)}][ERROR] camera_pub.py : {e}\n")
                    fp.write(f"[{str(now)}][WARNING] camera_pub.py : Image cannot be published. Camera pipeline restarting...\n")
                time.sleep(0.1)
                break

if __name__ == '__main__':
    while True:
        try:
            pub = rospy.Publisher('/fraw_img', Image, queue_size=10)
            rospy.init_node('camera_pub', anonymous=False)
            rate = rospy.Rate(30) #hz
            bridge = CvBridge()
            main()
        except rospy.ROSInterruptException:
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][WARNING] camera_pub.py : ROS Interrupted. Exiting...\n")
            break
        except KeyboardInterrupt:
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][WARNING] camera_pub.py : Keyboard Interruption... Exiting...\n")
            break
        except Exception as e:
            now = datetime.now()
            with open(f'/home/swarmnect/ros.log', 'a') as fp:
                fp.write(f"[{str(now)}][ERROR] camera_pub.py : {e}\n")
                fp.write(f"[{str(now)}][WARNING] camera_pub.py : Restarting...\n")
            time.sleep(0.1)
            
            