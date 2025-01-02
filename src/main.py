#!/usr/bin/env python3

# Satellite image classification node for ROS noetic.
# ---------------------------------------------------
# Description: 

import rospy
from std_msgs.msg import Float32
from geographic_msgs.msg import GeoPoint
from threading import Thread, Lock, Event
import datetime
from sentinelhub import(
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions,
    MosaickingOrder,
    get_utm_crs,
    transform_point,
    pixel_to_utm)
from geotypes import GeoBoundingBox, GeoCoordinate
from typing import List, Tuple
from time import sleep
from classifier import ImageClassifier
import numpy as np


class SentinelDownloader:
    def __init__(self) -> None:
        self.service_url = 'https://services.sentinel-hub.com'
        self.token_url = 'https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token'
        self.sh_client_id = 'a4f60b54-1651-45b6-859f-e0bbe68e946e'
        self.sh_client_secret = '3tkSRbFzn9JoFQr5stvlNXW63YOJy4PB'

        # Credentials profile
        self.profile = 'sentinel_data'

        # Initialization parameters
        self.RESOLUTION = 10
        self.SAT_IMG_RESOLUTION = 512

        self.date_range = 28
        self.today = datetime.date

        self.start_date = datetime.datetime(2024, 12, 1)
        self.end_date = datetime.datetime(2024, 12, 31)

        self.CRS = None

        self.config = self.get_config()

        self.true_color_image_evalscript =  """
                                            //VERSION=3
                                            function setup(){
                                                return{
                                                    input: ["B04", "B03", "B02"],
                                                    output: {
                                                        id: "default",
                                                        bands: 3,
                                                        sampleType: SampleType.FLOAT32
                                                    }
                                                };
                                            }

                                            function evaluatePixel(sample){
                                                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                                            }
                                            """

    def get_config(self) -> SHConfig:
        config = SHConfig()

        config.sh_base_url = self.service_url
        config.sh_token_url = self.token_url
        config.sh_client_id = self.sh_client_id
        config.sh_client_secret = self.sh_client_secret

        config.save(self.profile)

        return config

    def monitor_current_position(self, location_cache, stop_condition, classifier):
        rospy.loginfo("In image download thread.")
        image_count = 0

        while not stop_condition.is_set():
            current_position = location_cache.get()
            rospy.loginfo("In the loop")

            if current_position:
                print(f"Location from download thread {current_position.longitude}, {current_position.latitude}")

                if image_count < 2:
                    image_ndarray = self.download_sat_imagery(target_position_wgs84=(current_position.longitude, current_position.latitude))

                    uint8_image = image_ndarray.astype(np.uint8)

                    classifier.classify(uint8_image)

                    image_count += 1

            else:
                print(f"Position hasn't been updated")

            sleep(5)

    def get_image_tile_bounds_wgs84(self, center_coordinates_wgs84: Tuple[float]) -> List[GeoCoordinate]:
        image_corners = self.get_all_image_tile_corners(center_coordinates_wgs84)

        if image_corners:
            return [image_corners[0], image_corners[3]]

    def get_all_image_tile_corners(self, center_coordinates_wgs84: Tuple[float]) -> List[GeoCoordinate]:
        
        if abs(center_coordinates_wgs84[0]) > 0.0 and abs(center_coordinates_wgs84[1]) > 0.0:
            
            self.CRS = get_utm_crs(center_coordinates_wgs84[0], center_coordinates_wgs84[1])

            # Coordinates in UTM.
            center_coord_utm = transform_point((center_coordinates_wgs84[0], center_coordinates_wgs84[1]),
                                                CRS.WGS84,
                                                self.CRS)

            pixel_coordinates = pixel_to_utm(row= -self.SAT_IMG_RESOLUTION /2, 
                                            column= -self.SAT_IMG_RESOLUTION / 2,
                                            transform=[center_coord_utm[0], self.RESOLUTION, 0, center_coord_utm[1], 0, self.RESOLUTION])
        
            wgs84_transform = transform_point(pixel_coordinates, self.CRS, CRS.WGS84, True)
            wgs84_coordinates_top_left = GeoCoordinate(longitude= wgs84_transform[0], latitude= wgs84_transform[1])

            pixel_coordinates = pixel_to_utm(row= -self.SAT_IMG_RESOLUTION /2, 
                                            column= self.SAT_IMG_RESOLUTION / 2,
                                            transform=[center_coord_utm[0], self.RESOLUTION, 0, center_coord_utm[1], 0, self.RESOLUTION])

            wgs84_transform = transform_point(pixel_coordinates, self.CRS, CRS.WGS84, True)
            wgs84_coordinates_top_right = GeoCoordinate(longitude= wgs84_transform[0], latitude= wgs84_transform[1])

            pixel_coordinates = pixel_to_utm(row= self.SAT_IMG_RESOLUTION /2, 
                                            column= -self.SAT_IMG_RESOLUTION / 2,
                                            transform=[center_coord_utm[0], self.RESOLUTION, 0, center_coord_utm[1], 0, self.RESOLUTION])

            wgs84_transform = transform_point(pixel_coordinates, self.CRS, CRS.WGS84, True)
            wgs84_coordinates_bottom_left = GeoCoordinate(longitude= wgs84_transform[0], latitude= wgs84_transform[1])


            pixel_coordinates = pixel_to_utm(row= self.SAT_IMG_RESOLUTION /2, 
                                            column= self.SAT_IMG_RESOLUTION / 2,
                                            transform=[center_coord_utm[0], self.RESOLUTION, 0, center_coord_utm[1], 0, self.RESOLUTION])

            wgs84_transform = transform_point(pixel_coordinates, self.CRS, CRS.WGS84, True)
            wgs84_coordinates_bottom_right = GeoCoordinate(longitude= wgs84_transform[0], latitude= wgs84_transform[1])

            corner_coordinates = [wgs84_coordinates_top_left, 
                                wgs84_coordinates_top_right,
                                wgs84_coordinates_bottom_left,
                                wgs84_coordinates_bottom_right]

            return corner_coordinates

        else:
            # Returns an empty list.
            return list()

    def download_sat_imagery(self, target_position_wgs84: Tuple[float, float]):    # Lng, Lat

        roi_bbox_wgs84 = self.get_image_tile_bounds_wgs84(target_position_wgs84)

        sentinel_bbox = BBox(bbox=(roi_bbox_wgs84[0].longitude, 
                                    roi_bbox_wgs84[0].latitude, 
                                    roi_bbox_wgs84[1].longitude, 
                                    roi_bbox_wgs84[1].latitude), 
                            crs= CRS.WGS84)

        satellite_img_size = bbox_to_dimensions(sentinel_bbox, resolution=self.RESOLUTION)

        true_color_image_request = SentinelHubRequest(
                                                        data_folder='/home/scctower1/imagery',
                                                        evalscript= self.true_color_image_evalscript,
                                                        input_data= [SentinelHubRequest.input_data(
                                                            data_collection=DataCollection.SENTINEL2_L1C,
                                                            time_interval=(self.start_date, self.end_date),
                                                            mosaicking_order=MosaickingOrder.LEAST_CC,
                                                        )],
                                                        responses=[SentinelHubRequest.output_response(
                                                            "default", MimeType.TIFF)],
                                                        bbox=sentinel_bbox,
                                                        size=satellite_img_size,
                                                        config=self.config
                                                        )
        
        true_color_sat_img = true_color_image_request.get_data(save_data=True)

        image_data = true_color_sat_img[0]

        print(f"First pixel data {image_data[0, 0, :]} and {image_data.dtype}")

        return true_color_sat_img[0]

class GeoLocationCache:
    def __init__(self):
        self.mutex = Lock()
        self.cache = None
    
    
    def set(self, coordinate: GeoCoordinate):
        self.mutex.acquire()

        try:
            self.cache = coordinate

        finally:
            self.mutex.release()

    def get(self):
        self.mutex.acquire()

        try:
            return self.cache
        
        finally:
            self.mutex.release()

class GeoInfo:
    def __init__(self):
        self.position = None        

    def decode_position(self, message_response, args):
        if not rospy.is_shutdown():

            # print(f"Position: longitude{self.current_longitude} and latitude {self.current_latitude}")

            rospy.loginfo("Position lat %.8f,: lon %.8f", message_response.latitude, message_response.longitude)

            self.position = GeoCoordinate(longitude=message_response.longitude,
                                            latitude=message_response.latitude)

            location_cache = args[0]
            previous_position = location_cache.get()
            location_cache.set(self.position)

            print(f"Position updated")

            if self.position and previous_position:
                if not(previous_position.longitude == self.position.longitude) and not(previous_position.latitude == self.position.latitude):
                    location_cache.set(self.position)
                    print(f"New position isn't same as before")

    def get_position_messages(self, geo_coordinate_cache):
        sub_position_msg = rospy.Subscriber('geoposition',
                                        GeoPoint,
                                        self.decode_position,
                                        (geo_coordinate_cache,)) 


def main() -> None:
    # ROS node start
    rospy.init_node(name='SatelliteImgProcessingNode',
                    anonymous=False)

    stop_event = Event()

    # Constatntly check the GPS location. Check "/GPS" message.
    geo_location_cache = GeoLocationCache()

    rosGeoInfo = GeoInfo()
    position_listener_thread = Thread(target=rosGeoInfo.get_position_messages, 
                                        args=(geo_location_cache,))
    position_listener_thread.start()

    # Image classifier
    classifier_obj = ImageClassifier()
    classifier_obj.load_saved_model()

    # Satellite imagery downloading thread
    sat_img_download = SentinelDownloader()

    
    sat_img_acquiring_thread = Thread(target=sat_img_download.monitor_current_position,
                                        args=(geo_location_cache, stop_event, classifier_obj))
    sat_img_acquiring_thread.start()



    rate = rospy.Rate(10)

    rospy.spin()

    while not stop_event.is_set():

        if KeyboardInterrupt:
            stop_event.set()

            sleep(10)


    # Start satellite image download thread.


if __name__ == "__main__":
    main()
