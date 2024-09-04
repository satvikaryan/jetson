#!/usr/bin/env python3

import os
import subprocess
import time
import threading

import rospy
from sensor_msgs.msg import Image
from rose_tvmc_msg.msg import LEDControl

import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge, CvBridgeError


# List of ROS launch files you want to potentially execute
LAUNCH_FILES = [
    '/home/jetson/startup/launch/straight.launch',
    '/home/jetson/startup/launch/gate.launch',
    '/home/jetson/startup/launch/test.launch'
]

# The ROS image topic to subscribe to
IMAGE_TOPIC = '/oak/rgb/image_raw'


class ArucoStartup:
    def __init__(self, launch_files, image_topic, aruco_dict=aruco.DICT_APRILTAG_16h5):
        assert len(launch_files) > 0, "Launch files list cannot be empty"

        rospy.init_node("aruco_startup")

        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.launch_files = launch_files
        self.image_topic = image_topic
        self.launch_executed = False
        self.bridge = CvBridge()
        self.subscriber = None
        self.led_publisher = None
        self.frame = None
        self.frame_no = 0
        self.last_frame = -1

    def start(self):
        """
        Starts the ROS subscriber to listen on the specified image topic.
        """

        self.subscriber = rospy.Subscriber(
            self.image_topic, Image, self.on_frame, queue_size=1
        )

        self.led_publisher = rospy.Publisher(
            "/diagnostics/led", LEDControl, queue_size=10
        )

        self.led_on = False

        self.led_message = LEDControl()

        self.led_message.led = 0
        self.led_message.R = 0
        self.led_message.G = -1
        self.led_message.B = -1

        time.sleep(1)

        self.led_publisher.publish(
             self.led_message
        )

        print("Starting startup script.")

    def on_frame(self, msg):
        """
        Callback function for image topic messages. Converts ROS images to OpenCV format,
        processes them, and checks for ArUco markers.
        """
        if not self.led_on:
             self.led_message.R = -1
             self.led_message.G = -1
             self.led_message.B = 0
             self.led_publisher.publish(self.led_message)

             self.led_on = True

        if self.launch_executed:
            # Unsubscribe to stop processing further frames
            self.subscriber.unregister()
            print("Launch executed, stopped processing frames.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Detect markers in the current frame
        self.frame = cv_image
        self.frame_no += 1 

        self.frame_thread = threading.Thread(target=self.detect_in_frame, daemon=True)
        self.frame_thread.start()

    def detect_in_frame(self):
        while True:
            while not self.last_frame < self.frame_no or self.frame is None:
                time.sleep(0)
                print(self.last_frame, self.frame_no)
            
            self.last_frame = self.frame_no
            frame = self.frame

            processed_frame = self.preprocess_image(frame)
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(processed_frame)
            
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)

            if ids is not None and len(ids) > 0:
                marker_id = ids[0][0]
                if marker_id < len(self.launch_files):
                    self.launch_ros_file(marker_id)
                    self.launch_executed = True  # Prevent further executions
                else:
                    print(f"Marker ID {marker_id} does not correspond to a launch file.")
                return marker_id
            else:
                return 0
            


    def launch_ros_file(self, index):
        try:
            launch_file_path = self.launch_files[index]
            if os.path.exists(launch_file_path):
                print(f"Launching ROS file: {launch_file_path}")
                led_message = LEDControl()
                led_message.R = -1
                led_message.G = 0
                led_message.B = -1
                self.led_publisher.publish(led_message)
                # Use subprocess to launch the ROS launch file
                subprocess.Popen(["roslaunch", launch_file_path])
            else:
                print(f"Launch file does not exist: {launch_file_path}")
        except Exception as e:
            print(f"Failed to launch ROS file: {str(e)}")

    def preprocess_image(self, frame):
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(gray_frame)

        # Apply a Gaussian blur
        blurred_frame = cv2.GaussianBlur(clahe_frame, (0, 0), 20)

        # Sharpen image
        sharpened_frame = cv2.addWeighted(clahe_frame, 1.8, blurred_frame, -0.8, 0)

        return sharpened_frame

def main():
    # Create an instance of ArucoStartup with your launch files and image topic
    aruco_startup = ArucoStartup(launch_files=LAUNCH_FILES, image_topic=IMAGE_TOPIC)
    
    # Start the ArucoStartup object to begin subscribing to the image topic and processing frames
    aruco_startup.start()
    
    # Keep the script running until shutdown (e.g., Ctrl+C or shutdown signal from another part of your ROS system)
    rospy.spin()


if __name__ == "__main__":
    main()
