"""
ROS2 Data Logging Script for Prediction vs. Odometry Comparison

This script subscribes to predicted trajectory and odometry topics in a ROS2 system 
and logs the received data into a CSV file. The logged data includes timestamps, 
linear positions, orientations, velocities, and accelerations from both sources.

Key Features:
- Subscribes to prediction (`/base/prediction`) and odometry (`/crane/base/odometry`) topics.
- Logs data to a CSV file (`prediction_vs_odometry.csv`) with structured columns.
- Ensures time synchronization between prediction and odometry measurements.
- Uses buffered writing and periodic flushing to ensure data integrity.
- Gracefully handles node shutdown by closing the CSV file.

Dependencies:
- ROS2 (rclpy) for handling subscriptions and node execution.
- CSV and OS for file operations.
- Nav_msgs, Geometry_msgs, and Trajectory_msgs for message parsing.
- Datetime and Time for timestamp handling.

Usage:
- Ensure the necessary ROS2 topics are publishing data.
- Run this script in a ROS2 environment.
- The data will be continuously logged until the node is manually stopped.
"""

import rclpy
from rclpy.node import Node
import csv
import os
import rosbag2_py
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from rclpy.serialization import deserialize_message
from datetime import datetime
import numpy as np
import time
import pandas as pd
import logging


def prediction_to_csv(pred_topic_name, bag_file, output_csv):
    print('Starting to parse prediction values')
    reader = rosbag2_py.SequentialReader() # Reader created
    storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    data_dict = {}
    nhorizon = 90

    # Prepare to write csv

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        headers = ["timestamp_s"]
        for i in range(nhorizon):   # 90 is the preselected horizon step
            headers.extend([
                f"Pred_X_{i}", f"Pred_Y_{i}", f"Pred_Z_{i}",
                f"Pred_Qw_{i}", f"Pred_Qx_{i}", f"Pred_Qy_{i}", f"Pred_Qz_{i}",
                f"Pred_linVelX_{i}", f"Pred_linVelY_{i}", f"Pred_linVelZ_{i}",
                f"Pred_angVelX_{i}", f"Pred_angVelY_{i}", f"Pred_angVelZ_{i}",
                f"Pred_linAccX_{i}", f"Pred_linAccY_{i}", f"Pred_linAccZ_{i}",
                f"Pred_angAccX_{i}", f"Pred_angAccY_{i}", f"Pred_angAccZ_{i}"
            ])
        csv_writer.writerow(headers)

        # Iterate through messages
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == pred_topic_name:
                prediction_msg = deserialize_message(data, MultiDOFJointTrajectory)
                timestamp_s = timestamp / 1e9  # Convert to seconds

                if timestamp_s not in data_dict:
                    data_dict[timestamp_s] = {
                        "positions": [], "quaternions": [], 
                        "lin_vels": [], "ang_vels": [], 
                        "lin_accs": [], "ang_accs": []
                    }

                for step, point in enumerate(prediction_msg.points):
                    data_dict[timestamp_s]["positions"].append([
                        point.transforms[0].translation.x,
                        point.transforms[0].translation.y,
                        point.transforms[0].translation.z
                    ])
                    data_dict[timestamp_s]["quaternions"].append([
                        point.transforms[0].rotation.w,
                        point.transforms[0].rotation.x,
                        point.transforms[0].rotation.y,
                        point.transforms[0].rotation.z
                    ])
                    data_dict[timestamp_s]["lin_vels"].append([
                        point.velocities[0].linear.x,
                        point.velocities[0].linear.y,
                        point.velocities[0].linear.z
                    ])
                    data_dict[timestamp_s]["ang_vels"].append([
                        point.velocities[0].angular.x,
                        point.velocities[0].angular.y,
                        point.velocities[0].angular.z
                    ])
                    data_dict[timestamp_s]["lin_accs"].append([
                        point.accelerations[0].linear.x,
                        point.accelerations[0].linear.y,
                        point.accelerations[0].linear.z
                    ])
                    data_dict[timestamp_s]["ang_accs"].append([
                        point.accelerations[0].angular.x,
                        point.accelerations[0].angular.y,
                        point.accelerations[0].angular.z
                    ])

        for timestamp, values in sorted(data_dict.items()):
            row = [timestamp]

            for i in range(nhorizon):
                # Directly access the values without checking if i < len
                pos = values["positions"][i]
                row.extend([pos[0], pos[1], pos[2]])  # Pred_X_i, Pred_Y_i, Pred_Z_i

                quat = values["quaternions"][i]
                row.extend([quat[0], quat[1], quat[2], quat[3]])  # Pred_Qw_i, Pred_Qx_i, Pred_Qy_i, Pred_Qz_i

                lin_vel = values["lin_vels"][i]
                row.extend([lin_vel[0], lin_vel[1], lin_vel[2]])  # Pred_linVelX_i, Pred_linVelY_i, Pred_linVelZ_i

                ang_vel = values["ang_vels"][i]
                row.extend([ang_vel[0], ang_vel[1], ang_vel[2]])  # Pred_angVelX_i, Pred_angVelY_i, Pred_angVelZ_i

                lin_acc = values["lin_accs"][i]
                row.extend([lin_acc[0], lin_acc[1], lin_acc[2]])  

                ang_acc = values["ang_accs"][i]
                row.extend([ang_acc[0], ang_acc[1], ang_acc[2]]) 

            csv_writer.writerow(row)

    print(f"Converted prediction data from {bag_file} to {output_csv}")


def odometry_to_csv(odom_topic_name, bag_file, output_csv):
    # Create a reader for the rosbag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    # Prepare to write CSV for odometry data
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Time",
            "Odom_lin_X", "Odom_lin_Y", "Odom_lin_Z", "Odom_Qw", "Odom_Qx", "Odom_Qy", "Odom_Qz",
            "Odom_lin_velX", "Odom_lin_velY", "Odom_lin_velZ", "Odom_ang_velX", "Odom_ang_velY", "Odom_ang_velZ"
        ])

        # Iterate through the messages in the bag file
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            timestamp_s = timestamp / 1e9  # Convert timestamp to seconds
            
            # If it's an odometry message, store it in the odom_data dictionary
            if topic == odom_topic_name:
                odom_msg = deserialize_message(data, Odometry)
                Odom_lin_X = odom_msg.pose.pose.position.x
                Odom_lin_Y = odom_msg.pose.pose.position.y
                Odom_lin_Z = odom_msg.pose.pose.position.z
                Odom_Qw = odom_msg.pose.pose.orientation.w
                Odom_Qx = odom_msg.pose.pose.orientation.x
                Odom_Qy = odom_msg.pose.pose.orientation.y
                Odom_Qz = odom_msg.pose.pose.orientation.z
                Odom_lin_velX = odom_msg.twist.twist.linear.x
                Odom_lin_velY = odom_msg.twist.twist.linear.y
                Odom_lin_velZ = odom_msg.twist.twist.linear.z
                Odom_ang_velX = odom_msg.twist.twist.angular.x
                Odom_ang_velY = odom_msg.twist.twist.angular.y
                Odom_ang_velZ = odom_msg.twist.twist.angular.z

                # Write the data to the odometry CSV file
                csv_writer.writerow([
                    timestamp_s,  # Time in seconds
                    Odom_lin_X, Odom_lin_Y, Odom_lin_Z,
                    Odom_Qw, Odom_Qx, Odom_Qy, Odom_Qz,
                    Odom_lin_velX, Odom_lin_velY, Odom_lin_velZ,
                    Odom_ang_velX, Odom_ang_velY, Odom_ang_velZ
                ])
    print(f"Converted odometry data from {bag_file} to {output_csv}")

if __name__ == '__main__':

    INPUT_BAG_FLDR = '/home/subuntu/ROSBAGS/my_recording/'
    INPUT_BAG_FILE = 'my_recording_0'
    rclpy.init()

    # Call the prediction and odometry functions separately
    prediction_to_csv('/base/prediction', 
                      INPUT_BAG_FLDR + INPUT_BAG_FILE + '.db3',
                      'prediction_data.csv')

    odometry_to_csv('/crane/base/odometry', 
                    INPUT_BAG_FLDR + INPUT_BAG_FILE + '.db3',
                    'odometry_data.csv')