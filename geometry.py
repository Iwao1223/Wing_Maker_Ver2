import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
import math


class Geometry:
    @staticmethod
    def _offset_curve(fx, offset, sign, num_points=500):
        x_min, x_max = fx.x.min(), fx.x.max()
        x = np.linspace(x_min, x_max, num_points)
        y = fx(x)
        dy_dx = np.gradient(y, x)
        norm = np.sqrt(1 + dy_dx ** 2)
        nx = -dy_dx / norm
        ny = 1 / norm
        x_offset = x + sign * offset * nx
        y_offset = y + sign * offset * ny
        return interpolate.interp1d(x_offset, y_offset, kind='cubic')

    @staticmethod
    def offset_airfoil(fx_top, fx_bottom, offset, num_points=500):
        offset_fx_top = Geometry._offset_curve(fx_top, offset, -1, num_points)
        offset_fx_bottom = Geometry._offset_curve(fx_bottom, offset, 1, num_points)
        return [offset_fx_top, offset_fx_bottom]

    @staticmethod
    def trigonometry(x, y):
        length = 0
        for i in range(len(x) - 1):
            delta_l = 0
            delta_l = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
            length += delta_l
        return length

    @staticmethod
    def rotator(x, y, r):
        X = x * math.cos(r) - y * math.sin(r)
        Y = x * math.sin(r) + y * math.cos(r)
        return X, Y

    @staticmethod
    def find_intersection(fx, bottom_fx, x_min=None, x_max=None):
        if x_min is None:
            x_min = fx.x.min()
        if x_max is None:
            x_max = fx.x.max()
        def diff(x):
            return abs(fx(x)) - abs(bottom_fx(x))
        intersection_x = brentq(diff, x_min, x_max)
        return intersection_x

    @staticmethod
    def to_tuple_list(x_list, y_list):
        return list(zip(x_list, y_list))

    @staticmethod
    def find_intersection_circle_curve(fx, cx, cy, r, upper=True):
        x_min = cx - r + 1
        x_max = cx  # Avoid numerical issues at the edge

        def diff(x):
            val = r ** 2 - (x - cx) ** 2
            if val < 0:
                return np.nan
            if upper:
                y_circle = cy + np.sqrt(val)
            else:
                y_circle = cy - np.sqrt(val)
            return fx(x) - y_circle

        x_intersect = brentq(diff, x_min, x_max)
        y_intersect = fx(x_intersect)
        return x_intersect, y_intersect

    @staticmethod
    def arc_y(cx, cy, r, x, upper=True):
        """
        円弧のy座標を返す関数
        cx, cy: 円の中心座標
        r: 半径
        x: x座標（floatまたは配列）
        upper: Trueなら上側、Falseなら下側
        """
        val = r ** 2 - (x - cx) ** 2
        # 定義域外はnp.nan
        y = np.full_like(x, np.nan, dtype=float)
        mask = val >= 0
        if upper:
            y[mask] = cy + np.sqrt(val[mask])
        else:
            y[mask] = cy - np.sqrt(val[mask])
        return y

    @staticmethod
    def find_section_index(position_x, section_list):
        for j in range(len(section_list)):
            if position_x <= section_list[j]:
                return j - 1
        return None
