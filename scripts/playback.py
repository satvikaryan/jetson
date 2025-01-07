#!/usr/bin/env python3
import csv
import rospy
from std_msgs.msg import Int32MultiArray
from time import sleep

CSV_FILE = "pwm_log.csv"  
PWM_TOPIC = "/control/pwm" 


def replay_pwm():
    """
    Reads PWM values from a CSV file and publishes them to the AUV.
    """
    rospy.init_node("pwm_replay", anonymous=True)
    pwm_publisher = rospy.Publisher(PWM_TOPIC, Int32MultiArray, queue_size=10)

    try:
        with open(CSV_FILE, mode="r") as file:
            reader = csv.reader(file)
            header = next(reader)  

            start_time = None
            for row in reader:
                
                timestamp, pwm_values = float(row[0]), row[1]
                pwm_values = list(map(int, pwm_values.strip("[]").split(",")))

                if start_time is None:
                    start_time = timestamp
                sleep_duration = timestamp - start_time
                start_time = timestamp

                rospy.loginfo(f"Replaying PWM: {pwm_values} (After {sleep_duration}s)")
                sleep(sleep_duration)

                msg = Int32MultiArray(data=pwm_values)
                pwm_publisher.publish(msg)
    except FileNotFoundError:
        rospy.logerr(f"CSV file '{CSV_FILE}' not found. Aborting replay.")
    except rospy.ROSInterruptException:
        rospy.loginfo("PWM replay interrupted.")
    finally:
        rospy.loginfo("PWM replay completed or terminated.")


if __name__ == "__main__":
    replay_pwm()
