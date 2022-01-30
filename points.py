from ctypes import *
import random
helpers = CDLL('./helper.so')


class Point(Structure):
    _fields_ = [('x', c_float), ('y', c_float)]

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Setting_1():
    w = 800
    h = 800
    number_of_points = 1000
    min_border_distance = 0.01
    border_weight = 250
    learning_rate = 800
    repel_weight = 150
    max_magnitude = 6
    point_radius = 3


class Setting_2():
    w = 800
    h = 800
    number_of_points = 20
    min_border_distance = 0.01
    border_weight = 1
    learning_rate = 100000
    repel_weight = 20
    max_magnitude = 6
    point_radius = 6


setting = Setting_1()


def initialize_points():
    return (Point * setting.number_of_points)(*[Point(
        random.uniform(setting.min_border_distance,
                       setting.w - setting.min_border_distance),
        random.uniform(setting.min_border_distance,
                       setting.h - setting.min_border_distance)
    ) for i in range(setting.number_of_points)])


def update_points(points, mouse_pos, is_mouse_down):
    helpers.update_points(
        byref(points),
        (c_uint)(len(points)),
        (c_uint)(setting.w),
        (c_uint)(setting.h),
        (c_float)(setting.border_weight),
        (c_float)(setting.min_border_distance),
        (c_float)(setting.learning_rate),
        (c_float)(setting.max_magnitude),
        Point(mouse_pos[0], mouse_pos[1]),
        (c_int)(is_mouse_down),
        (c_float)(setting.repel_weight)
    )
