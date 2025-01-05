#!/usr/bin/env python3

import geopandas as gpd
import cv2
import numpy as np
from PIL import Image
from sentinelhub import transform_point, pixel_to_utm, get_utm_crs, CRS
from geographic_msgs.msg import MapFeature, KeyValue
import rospy
import uuid
from uuid_msgs.msg import UniqueID
from std_msgs.msg import Header
import json


class ShapeFileGenerator:
    def __init__(self) -> None:
        self.binary_images = {}
        self.img_height = None
        self.img_width = None

        self.ros_map_publisher = None

    def classification_to_binary(self, classification_result_img, btm_left_coordinates):
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
                self.binary_to_shapevector(binary_img_layer, object_class, btm_left_coordinates) 


    def binary_to_shapevector(self, binary_img, object_class, btm_left_coordinates):

        print(f"Image type {binary_img.dtype} and shape {binary_img.shape}")
        # values = cv2.findContours(binary_img.astype(dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(f"Velus to unpack {values}").astype(dtype=np.uint8)

        _, binary_thresh = cv2.threshold(binary_img, 128, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Use image open/close operations to remove small parts from the image.

        print(f"Number of contours {len(contours)}")

        CRS = get_utm_crs(btm_left_coordinates.longitude, btm_left_coordinates.latitude)

        # Coordinates in UTM.
        btm_left_utm = transform_point((btm_left_coordinates.longitude, btm_left_coordinates.latitude),
                                            CRS.WGS84, CRS)

        # Polygons list
        polygons = []

        for contour in contours:
            points_array = []
            geo_coord_array = []
            for point in contour:
                pixel_coordinates = pixel_to_utm(row= (512 -point[0][1]), 
                                column= point[0][0],
                                transform=[btm_left_utm[0], 10, 0, btm_left_utm[1], 0, 10])
        
                wgs84_transform = transform_point(pixel_coordinates, CRS, CRS.WGS84, True)
                points_array.append((point[0][0], (512 - point[0][1])))
                geo_coord_array.append([wgs84_transform[1], wgs84_transform[0]])
            
            polygon = MapFeature()
            
            # coords = [[59.6 , 24.6], [59.2, 24.0], [59.4, 23.5]]
            # points_dict = {"type" : "polygon",
            #                 "vertices": geo_coord_array} 
            json_string = json.dumps(geo_coord_array)

            polygon.components.append(self.create_unique_id())


            polygon.props.append(KeyValue(key="type", value="polygon"))
            polygon.props.append(KeyValue(key="vertices", value=json_string))

            self.publish_ice_map_layers(polygon)

            print(f"Points array {points_array[0]}")

    def create_unique_id(self):
        return UniqueID(uuid=list(uuid.uuid4().bytes))
        
    def set_ros_topics(self, ros_publisher):
        self.ros_map_publisher = ros_publisher

    def publish_ice_map_layers(self, data):
        self.ros_map_publisher.publish(data)
