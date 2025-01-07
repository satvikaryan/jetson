#!/usr/bin/env python3
import rospy
import csv
from std_msgs.msg import Int32MultiArray

class PWMRecorder:
    def __init__(self):
        rospy.init_node("pwm_recorder", anonymous=True)
        self.csv_file = open("pwm_data.csv", mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "pwm_values"]) 

        rospy.Subscriber("/control/pwm", Int32MultiArray, self.callback)
        rospy.loginfo("PWM Recorder node initialized. Recording PWM data to 'pwm_data.csv'.")

    def callback(self, msg):
        timestamp = rospy.Time.now().to_sec()
        self.csv_writer.writerow([timestamp, list(msg.data)])
        rospy.loginfo(f"Recorded PWM: {msg.data}")

    def shutdown(self):
        self.csv_file.close()
        rospy.loginfo("CSV file closed. Shutting down PWM Recorder node.")

if __name__ == "__main__":
    recorder = PWMRecorder()
    rospy.on_shutdown(recorder.shutdown)
    rospy.spin()
