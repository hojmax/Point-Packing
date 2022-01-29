from ctypes import *
import random
helpers = CDLL('./helper.so')


class Point(Structure):
    _fields_ = [('x', c_float), ('y', c_float)]

    def __init__(self, x, y):
        self.x = x
        self.y = y


w = 800
h = 800
number_of_points = 1000
min_border_distance = 0.01
border_weight = 250
learning_rate = 1000
# This translates to a point moving a maximum of 4 pixels per update
max_magnitude = 4


def initialize_points():
    return (Point * number_of_points)(*[Point(
        random.uniform(min_border_distance, w - min_border_distance),
        random.uniform(min_border_distance, h - min_border_distance)
    ) for i in range(number_of_points)])


def update_points(points):
    helpers.update_points(
        byref(points),
        (c_uint)(len(points)),
        (c_int)(w),
        (c_int)(h),
        (c_int)(border_weight),
        (c_float)(min_border_distance),
        (c_float)(learning_rate),
        (c_float)(max_magnitude),
    )
