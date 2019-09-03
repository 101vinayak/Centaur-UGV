import rospy
from std_msgs.msg import Int16, UInt8MultiArray
from threading import Thread

FORWARD = 1
BACKWARD = 0
SLOW = 100
FAST = 200

def update_angle(new_angle):
	angle = int(new_angle.data)

def start_fetching():
	rospy.init_node('Navigate', anonymous=True)
	rospy.Subscriber('IMU_Output', Int16, update_angle)

	global pub
	pub = rospy.Publisher('MCU_input', std_msgs.msg.UInt8MultiArray)

def motor_cmd(left_dir, left_speed, right_dir, right_speed):
	pub.publish(UInt8MultiArray(4, [left_dir, left_speed, right_dir, right_speed]))

def turn_left(target_angle):
	while angle < target_angle:
		motor_cmd(FORWARD, 0, FORWARD, SLOW)

def turn_right(target_angle):
	while angle > target_angle:
		motor_cmd(FORWARD, SLOW, FORWARD, 0)

def move_forward(speed):
	motor_cmd(FORWARD, speed, FORWARD, speed)

def move_backward(speed):
	motor_cmd(BACKWARD, speed, BACKWARD, speed)

def turn(target_angle):
	"""Negative angle = right turn
	   and positive angle = left turn"""

	if angle < target_angle:
		turn_left(target_angle)
	else:
		turn_right(target_angle)

def init():
	global angle
	angle = 0

	t = Thread(target=start_fetching)
	t.start()

if __name__ == '__main__':
	init()