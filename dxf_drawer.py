# dxf_drawer.py
import ezdxf
import numpy as np

class DXFDrawer:

    def __init__(self):
        self.doc = ezdxf.new('R2018', setup=True)
        self.msp = self.doc.modelspace()


    def draw_polyline(self, points):
        self.msp.add_lwpolyline(points, format="xy", close=True)


    def draw_circle(self, center, diameter_list):
        if isinstance(diameter_list, np.ndarray):
            radius = diameter_list[0] / 2
        else:
            radius = diameter_list / 2
        self.msp.add_circle(center, radius=radius)
    def draw_ellipse(self, center, diameter_list, ellipse_dgeree):
        major_axis_x = diameter_list[0] / 2 * np.cos(np.deg2rad(ellipse_dgeree))
        major_axis_y = diameter_list[0] / 2 * np.sin(np.deg2rad(ellipse_dgeree))
        major_axis = (major_axis_x, major_axis_y)
        ratio = diameter_list[1] / diameter_list[0]
        self.msp.add_ellipse(center, major_axis=major_axis, ratio=ratio)

    def save(self, filename):
        self.doc.saveas(filename)