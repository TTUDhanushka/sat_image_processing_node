#!/usr/bin/env python3


class GeoCoordinate:
    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude


class GeoBoundingBox:
    def __init__(self, top_left: GeoCoordinate, bottom_right: GeoCoordinate) -> None:
        self.top_left = top_left
        self.bottom_right = bottom_right

