#!/usr/bin/env python3

import geopandas as gpd
import cv2
import numpy as np
from PIL import Image
from sentinelhub import transform_point
from geographic_msgs.msg import MapFeature, KeyValue
import rospy
import uuid
from uuid_msgs.msg import UniqueID
from std_msgs.msg import Header


class ShapeFileGenerator:
    def __init__(self) -> None:
        self.binary_images = {}
        self.img_height = None
        self.img_width = None

        self.ros_map_publisher = None

    def classification_to_binary(self, classification_result_img):
        self.img_height, self.img_width = classification_result_img.shape

        num_classes = classification_result_img.max() + 1

        print(f"Num classes in the result {num_classes}")

        for object_class in range(num_classes):
            binary_img_layer = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

            detected_pixel_count = 0

            for i in range(self.img_height):
                for j in range (self.img_width):

                    if classification_result_img[i, j] == object_class:
                        binary_img_layer[i, j] = 255
                        detected_pixel_count += 1
                    else:
                        binary_img_layer[i, j] = 0



            print(f"Class id: {object_class} and no of pixels {detected_pixel_count}")

            if detected_pixel_count > 0:
                pil_image = Image.fromarray(binary_img_layer.astype(dtype=np.uint8), 'L')

                pil_image.show()

                self.binary_images.update({object_class : binary_img_layer})
                self.binary_to_shapevector(binary_img_layer, object_class) 


    def binary_to_shapevector(self, binary_img, object_class):

        print(f"Image type {binary_img.dtype} and shape {binary_img.shape}")
        # values = cv2.findContours(binary_img.astype(dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(f"Velus to unpack {values}").astype(dtype=np.uint8)

        _, binary_thresh = cv2.threshold(binary_img, 128, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Use image open/close operations to remove small parts from the image.

        print(f"Number of contours {len(contours)}")

        # Polygons list
        polygons = []

        for contour in contours:
            points_array = []
            for point in contour:
                points_array.append((point[0][0], (512 - point[0][1])))

            polygon = MapFeature()
            

            polygon.components.append(self.create_unique_id())

            polygon.props.append(KeyValue(key="vertices", value = "[(1.0 , 1.0), (2.0, 2.0)]"))

            self.publish_ice_map_layers(polygon)

            print(f"Points array {points_array[0]}")

    def create_unique_id(self):
        return UniqueID(uuid=list(uuid.uuid4().bytes))
        
    def set_ros_topics(self, ros_publisher):
        self.ros_map_publisher = ros_publisher

    def publish_ice_map_layers(self, data):
        self.ros_map_publisher.publish(data)
