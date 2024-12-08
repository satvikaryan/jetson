#!/usr/bin/env python3

import rospy
import cv2
import depthai as dai
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OAKDVideoPublisher:
    def __init__(self):
        rospy.init_node('oakd_video_publisher', anonymous=True)
        
        # Publishers for different streams
        self.rgb_pub = rospy.Publisher('/oakd/rgb/image_raw', Image, queue_size=10)
        self.left_pub = rospy.Publisher('/oakd/left/image_raw', Image, queue_size=10)
        self.right_pub = rospy.Publisher('/oakd/right/image_raw', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/oakd/depth/image_raw', Image, queue_size=10)
        
        # CV Bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        # Create OAK-D pipeline
        self.pipeline = dai.Pipeline()
        
        # RGB Camera
        self.rgb_cam = self.pipeline.create(dai.node.ColorCamera)
        self.rgb_cam.setPreviewSize(416, 416)
        self.rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        self.rgb_cam.setInterleaved(False)
        
        # Stereo cameras
        self.left_cam = self.pipeline.create(dai.node.MonoCamera)
        self.right_cam = self.pipeline.create(dai.node.MonoCamera)
        
        self.left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        
        self.left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # Depth
        self.depth = self.pipeline.create(dai.node.StereoDepth)
        self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        
        # Linking
        self.left_cam.out.link(self.depth.left)
        self.right_cam.out.link(self.depth.right)
        
    def run(self):
        with dai.Device(self.pipeline) as device:
            # Get output queues
            rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
            right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            while not rospy.is_shutdown():
                # RGB Frame
                rgb_frame = rgb_queue.get()
                if rgb_frame is not None:
                    rgb_img = rgb_frame.getCvFrame()
                    rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
                    self.rgb_pub.publish(rgb_msg)
                
                # Left Mono Frame
                left_frame = left_queue.get()
                if left_frame is not None:
                    left_img = left_frame.getCvFrame()
                    left_msg = self.bridge.cv2_to_imgmsg(left_img, "mono8")
                    self.left_pub.publish(left_msg)
                
                # Right Mono Frame
                right_frame = right_queue.get()
                if right_frame is not None:
                    right_img = right_frame.getCvFrame()
                    right_msg = self.bridge.cv2_to_imgmsg(right_img, "mono8")
                    self.right_pub.publish(right_msg)
                
                # Depth Frame
                depth_frame = depth_queue.get()
                if depth_frame is not None:
                    depth_img = depth_frame.getCvFrame()
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_img, "32FC1")
                    self.depth_pub.publish(depth_msg)
                
                rospy.Rate(30).sleep()  # 30 Hz publishing rate

if __name__ == '__main__':
    try:
        publisher = OAKDVideoPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass