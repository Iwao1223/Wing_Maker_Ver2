import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


class WingInterpolate:
    def __init__(self, config):
        self.config = config

    def interpolate(self):
        airfoil_mix_gradient = 100 / self.config.brock_span
        chord_gradient = (self.config.chord_T - self.config.chord_R) / self.config.brock_span
        spar_position_gradient = ((self.config.spar_position_T - self.config.spar_position_R)
                                  / self.config.brock_span)
        rear_spar_position_gradient = ((self.config.rear_spar_position_T - self.config.rear_spar_position_R)
                                       / self.config.brock_span)
        spar_major_diameter_gradient = ((self.config.spar_major_diameter_T - self.config.spar_major_diameter_R)
                                  / self.config.brock_span)
        spar_minor_diameter_gradient = ((self.config.spar_minor_diameter_T - self.config.spar_minor_diameter_R)
                                  / self.config.brock_span)
        rear_spar_diameter_gradient = ((self.config.rear_spar_diameter_T - self.config.rear_spar_diameter_R)
                                       / self.config.brock_span)

        chord_list = []
        mix_list = []
        spar_position_list = []
        rear_spar_position_list = []
        spar_major_diameter_list = []
        spar_minor_diameter_list = []
        rear_spar_diameter_list = []
        spar_diameter_list = []
        ellipse_ratio_list = []
        for i in self.config.rib_station_list:
            chord = self.config.chord_R + chord_gradient * i
            mix = airfoil_mix_gradient * i
            spar_position = self.config.spar_position_R + spar_position_gradient * i
            rear_spar_position = self.config.rear_spar_position_R + rear_spar_position_gradient * i
            spar_major_diameter = self.config.spar_major_diameter_R + spar_major_diameter_gradient * i
            spar_minor_diameter = self.config.spar_minor_diameter_R + spar_minor_diameter_gradient * i
            rear_spar_diameter = self.config.rear_spar_diameter_R + rear_spar_diameter_gradient * i

            chord_list.append(chord)
            mix_list.append(mix)
            spar_position_list.append(spar_position)
            rear_spar_position_list.append(rear_spar_position)
            spar_major_diameter_list.append(spar_major_diameter)
            spar_minor_diameter_list.append(spar_minor_diameter)
            spar_diameter_list.append([spar_major_diameter, spar_minor_diameter])
            rear_spar_diameter_list.append(rear_spar_diameter)
        chord_list = np.array(chord_list)
        mix_list = np.array(mix_list)
        spar_position_list = np.array(spar_position_list)
        rear_spar_position_list = np.array(rear_spar_position_list)
        spar_diameter_list = np.array(spar_diameter_list)
        rear_spar_diameter_list = np.array(rear_spar_diameter_list)
        if self.config.circle_or_ellipse == 'circle':
            ellipse_ratio_list = None
        elif self.config.circle_or_ellipse == 'ellipse':
            ellipse_gradient = (self.config.ellipse_degree_T - self.config.ellipse_degree_R) / self.config.brock_span
            for i in self.config.rib_station_list:
                ellipse_ratio_list.append(self.config.ellipse_degree_R + ellipse_gradient * i)
            ellipse_ratio_list = np.array(ellipse_ratio_list)

        return chord_list, mix_list, spar_position_list, rear_spar_position_list, spar_diameter_list, rear_spar_diameter_list , ellipse_ratio_list
