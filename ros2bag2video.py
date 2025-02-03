#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Bey Hao Yun.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import sys
import cv2
import rclpy
import rclpy.executors
import getopt
import subprocess
import threading
import queue
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage

try:
    from theora_image_transport.msg import Packet
except Exception:
    pass

VIDEO_CONVERTER_TO_USE = "ffmpeg"
# Sentinel value: when this is put into the queue, the image writer thread will terminate.
SENTINEL = None


def print_help():
    """
    Prints usage help.
    """
    print(
        "ros2bag2video.py [--fps 25] [--rate 1] [-o outputfile] [-v] "
        + "[-s] [-t topic] bagfile1 [bagfile2] ..."
    )
    print()
    print(
        "Converts image sequence(s) in ROS bag file(s) to video file(s)"
        + " with fixed frame rate using",
        VIDEO_CONVERTER_TO_USE,
    )
    print(VIDEO_CONVERTER_TO_USE, "needs to be installed!")
    print()
    print("--fps   Sets FPS value that is passed to", VIDEO_CONVERTER_TO_USE)
    print("        Default is 25.")
    print("-h      Displays this help.")
    print("--ofile (-o) sets output file name.")
    print(
        "        If no output file name (-o) is given the filename"
        + " 'output.mp4' is used"
    )
    print(
        "        Multiple image topics are supported only when -o "
        + "option is _not_ used."
    )
    print(
        "        ",
        VIDEO_CONVERTER_TO_USE,
        " will guess the format according to the file extension.",
    )
    print(
        "        Compressed and raw image messages are supported with "
        + "mono8 and bgr8/rgb8/bggr8/rggb8 formats."
    )
    print("--rate  (-r) You may slow down or speed up the video.")
    print("        Default is 1.0, that keeps the original speed.")
    print(
        "-s      Displays each and every image extracted from the ROS bag file"
        + " (cv_bridge is needed)."
    )
    print(
        '--topic (-t) Only the images from the specified topic are used for'
        + " video output."
    )
    print("-v      Verbose messages are displayed.")


class ImageWriterThread(threading.Thread):
    """
    A thread that takes images from a queue and writes them to disk.
    When the SENTINEL is encountered, the thread will exit.
    """

    def __init__(self, image_queue):
        super().__init__()
        self.image_queue = image_queue

    def run(self):
        while True:
            item = self.image_queue.get()
            # Exit if a SENTINEL is received.
            if item is SENTINEL:
                self.image_queue.task_done()
                break
            frame_no, img = item
            filename = str(frame_no).zfill(4) + ".png"
            cv2.imwrite(filename, img)
            self.image_queue.task_done()


class RosVideoWriter(Node):
    """
    A ROS2 node that extracts images from a bag file and converts them into a video.
    The image saving process is offloaded to a separate thread to reduce load in the callback.
    """

    def __init__(self, args):
        super().__init__("ros2bag2videos")
        self.fps = 25
        self.rate = 1.0
        self.frame_no = 1
        self.opt_out_file = "output.mp4"
        self.opt_topic = ""
        self.opt_verbose = False
        self.bridge = CvBridge()
        self.pix_fmt = "yuv420p"
        self.msg_fmt = ""
        self.bag_file = None
        self.msgtype = None
        self.count = 0

        # Parse command line arguments.
        if len(args) < 2:
            print("Please specify a ROS2 bag file!")
            print_help()
            sys.exit(1)
        try:
            opt_files = self.parse_args(args[1:])
            if len(opt_files) < 1:
                print("Bag file not specified!")
                sys.exit(1)
            self.bag_file = opt_files[0]
            print("FPS (int) =", self.fps)
            print("Rate (float) =", self.rate)
            print("Topic (str) =", self.opt_topic)
            print("Output File (str) =", self.opt_out_file)
            print("Verbose (bool) =", self.opt_verbose)
        except getopt.GetoptError:
            print_help()
            sys.exit(2)

        # Retrieve bag file information.
        self._read_ros_bag_info()

        # Prepare a queue and thread for asynchronous image writing.
        self.image_queue = queue.Queue(maxsize=1000)
        self.writer_thread = ImageWriterThread(self.image_queue)
        self.writer_thread.start()

        # Create a subscription with an increased QoS depth for high-rate playback.
        qos_profile = rclpy.qos.QoSProfile(depth=400)
        self.subscription = self.create_subscription(
            self.msgtype, self.opt_topic, self.listener_callback, qos_profile
        )
        self.get_logger().info(
            f"Subscribed to {self.opt_topic} with msg type {self.msgtype}"
        )

        # Start ROS bag playback.
        self._play_process = self._playback_ros_bag()

    def _read_ros_bag_info(self):
        self.get_logger().info("Reading info from bag file: " + self.bag_file)
        with subprocess.Popen(
            ["ros2", "bag", "info", self.bag_file], stdout=subprocess.PIPE
        ) as proc:
            rosbag2_info = proc.stdout.read().decode("utf-8").splitlines()
        self.msgfmt_literal, self.count = self.get_topic_info(rosbag2_info)
        self.msgtype = self.filter_image_msgs(self.msgfmt_literal)
        self.get_logger().info(f"ROS Message name = {self.msgfmt_literal}")
        self.get_logger().info(f"Image count = {self.count}")
        self.get_logger().info(f"msgtype = {self.msgtype}")

    def _playback_ros_bag(self):
        self.get_logger().info("Starting ROS bag playback...")
        process = subprocess.Popen(
            [
                "ros2",
                "bag",
                "play",
                self.bag_file,
                "-r",
                str(self.rate),
                "--topics",
                self.opt_topic,
            ]
        )
        return process

    def parse_args(self, args):
        opts, opt_files = getopt.getopt(
            args, "hsvr:o:t:", ["fps=", "rate=", "ofile=", "topic="]
        )
        for opt, arg in opts:
            if opt == "-h":
                print_help()
                sys.exit(0)
            elif opt == "-v":
                self.opt_verbose = True
            elif opt in ("--fps"):
                self.fps = int(arg)
            elif opt in ("-r", "--rate"):
                self.rate = float(arg)
            elif opt in ("-o", "--ofile"):
                self.opt_out_file = arg
            elif opt in ("-t", "--topic"):
                self.opt_topic = arg
            else:
                print("Unknown option:", opt, "arg:", arg)

        if self.fps <= 0:
            self.get_logger().warn("Invalid fps provided, defaulting to 1")
            self.fps = 1

        if self.rate <= 0:
            self.get_logger().warn("Invalid rate provided, defaulting to 1")
            self.rate = 1

        return opt_files

    def filter_image_msgs(self, msgfmt_literal):
        if "sensor_msgs/msg/Image" == msgfmt_literal:
            return Image
        elif "sensor_msgs/msg/CompressedImage" == msgfmt_literal:
            return CompressedImage
        elif "theora_image_transport/msg/Packet" == msgfmt_literal:
            return Packet
        else:
            self.get_logger().error("Unsupported message type: " + msgfmt_literal)
            sys.exit(1)

    def get_topic_info(self, rosbag2_info):
        msgtype = ""
        count = 0
        for line in rosbag2_info:
            if self.opt_topic in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("Type:"):
                        if i + 1 < len(parts):
                            msgtype = parts[i + 1]
                    if part.startswith("Count:"):
                        if i + 1 < len(parts):
                            try:
                                count = int(parts[i + 1])
                            except ValueError:
                                count = 0
        return msgtype, count

    def listener_callback(self, msg):
        # Log the receipt of the image.
        self.get_logger().info(
            f"Image Received [{self.frame_no}/{self.count}]"
        )

        # For simplicity, we fix the pixel format (yuv420p) and message format (rgb8)
        self.pix_fmt = "yuv420p"
        self.msg_fmt = "bgra8"

        # Convert the ROS image message to an OpenCV image using cv_bridge.
        try:
            if self.msgtype == CompressedImage:
                img = self.bridge.compressed_imgmsg_to_cv2(msg, self.msg_fmt)
            elif self.msgtype == Image:
                img = self.bridge.imgmsg_to_cv2(msg, self.msg_fmt)
            else:
                self.get_logger().error("Unsupported message type.")
                sys.exit(1)
        except Exception as e:
            self.get_logger().error("Error converting image: " + str(e))
            return

        # Immediately enqueue the image to avoid blocking in the callback.
        self.image_queue.put((self.frame_no, img))
        self.frame_no += 1

        # When the last frame is received, signal the writer thread to finish.
        if self.frame_no > self.count:
            self.image_queue.put(SENTINEL)
            # Wait until all images in the queue are processed.
            self.image_queue.join()
            self.writer_thread.join()

            self.get_logger().info("All images written, starting video creation...")

            # Use ffmpeg to create the video from the saved images.
            ffmpeg_cmd = [
                VIDEO_CONVERTER_TO_USE,
                "-framerate", str(self.fps),
                "-pattern_type", "glob",
                "-i", "*.png",
                "-c:v", "libx264",
                "-pix_fmt", self.pix_fmt,
                self.opt_out_file,
                "-y",
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
            except subprocess.CalledProcessError as e:
                self.get_logger().error("ffmpeg failed: " + str(e))
                sys.exit(1)
            # Remove individual image files.
            subprocess.call("rm *.png", shell=True)
            self.get_logger().info("Video creation complete. Exiting.")
            sys.exit(0)

    def exit(self, value):
        if self._play_process:
            self._play_process.kill()
        sys.exit(value)


def main(args=None):
    """
    The main function.
    Initializes the ROS2 node and spins it using a multi-threaded executor.
    """
    rclpy.init(args=args)
    videowriter = RosVideoWriter(sys.argv)
    try:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(videowriter)
        executor.spin()
    except KeyboardInterrupt:
        videowriter.get_logger().info(
            "KeyboardInterrupt received, shutting down gracefully."
        )
    finally:
        videowriter.destroy_node()


if __name__ == "__main__":
    main()
