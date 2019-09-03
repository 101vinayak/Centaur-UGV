#!/usr/bin/python
#This is the basic obstacle detection module

import matplotlib.pyplot as plt
import numpy as np
import rospy
from time import time
import os
from sensor_msgs.msg import LaserScan

# This program includes the following hardcoded values:
# Smallest cluster to detect as an object 
# Critical distance below which an object should be an obstacle

# The values are just placeholders for now. They are subject
# to change with testing

OBJECT_RESOLUTION = 0.03 # Size of the smallest object (in radians)
CRITICAL_DISTANCE = 1    # Set to zero to report all ojects as obstacles
LIDAR_RESOLUTION = 10 # In degrees 

def build_object(data, start_index, end_index):
	"""This function builds an obstacle object from the LIDAR data"""

	data_len = len(data.ranges)
	start_bound = (data_len/2 - start_index)*data.angle_increment
	end_bound = (data_len/2 - end_index)*data.angle_increment
	min_distance = np.min(data.ranges[start_index:end_index+1])

	if np.isnan(min_distance):
		# Faulty reading
		return None
	elif abs(start_bound - end_bound) < OBJECT_RESOLUTION:
		# Object too small
		return None


	return {'start_bound': float(start_bound),
			'end_bound': float(end_bound),
			'distance': float(min_distance),
			'start_index': float(start_index),
			'end_index': float(end_index)}

def get_obstacles(data, partitions, critical_distance=0):
	"""This function returns an array of objects 
	that are closer than critical_distance. 
	The array is sorted in increasing order of distance"""

	boundaries = np.argwhere(partitions)

	if len(boundaries) == 0:
		return []

	first_bound = boundaries[0]
	boundaries = boundaries[1:] # Remove the first_bound from the list

	obstacles = []
	prev_bound = first_bound
	for boundary in boundaries:
		obj = build_object(data, prev_bound, boundary)
		prev_bound = boundary

		if obj is None or (critical_distance > 0 and obj['distance'] >= critical_distance):
			continue
		else:
			obstacles.append(obj)

	obstacles.sort(key=lambda x: x['distance'])
	return obstacles

def is_obstructed(angle, obstacles):
	for obstacle in obstacles:
		if angle > obstacle['start_index'] and angle < obstacle['end_index']:
			return True
	return False

def plot_laser_data(data, partitions, obstacles, threats):
	"""This functions plots all the obtained information on a polar graph"""

	ax = plt.subplot(111, projection='polar')	
	plt.cla()
	theta = range(len(data.ranges))
	theta = [data.angle_min + data.angle_increment*t for t in theta]
	ax.scatter(theta, data.ranges, s = 2)

	#plt.plot(theta, data.ranges, '#c2c3c4')

	#plt.plot(theta, partitions, 'r')
	#plt.plot(theta, obstacles, 'g')

	res = np.deg2rad(LIDAR_RESOLUTION)/data.angle_increment
	obstructed = [0]*len(theta)
	free = [0]*len(theta)
	for i in range(0, len(theta), np.int(res)):
		if is_obstructed(i, threats):
			obstructed[i] = 3
		else:
			free[i] = 3

	plt.plot(theta, free, 'g')
	plt.plot(theta, obstructed, 'r')
	plt.draw()
	plt.pause(0.00000000000000000000000000000000000000000001)


def on_laser_scan(a_data):	
	data = a_data
	data.ranges = [data.range_max if np.isnan(x) else x for x in data.ranges]
	grads = np.gradient(np.array(data.ranges))
	partitions = [1 if abs(i) > CRITICAL_DISTANCE else 0 for i in grads]
	
	obstacles = get_obstacles(data, partitions, CRITICAL_DISTANCE)
	obstacle_graph = [0]*len(data.ranges)
	for ob in obstacles:
		index = (ob['start_index'] + ob['end_index'])/2
		obstacle_graph[int(index)] = 1

	plot_laser_data(data, partitions, obstacle_graph, obstacles)	

def main():
	plt.ion()
	
	rospy.init_node('what', anonymous=True)
	ret = rospy.Subscriber("scan", LaserScan, on_laser_scan)
	print "PRESS CTRL-C TO QUIT"
	rospy.spin()
	os._exit(0) # This line supresses the stacktrace produced by tkinter

if __name__ == '__main__':
	main()
