import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd


class AirfoilInterpolate:
    def __init__(self,airfoil_df_list,chord_list):
        self.airfoil_df_list = airfoil_df_list
        self.chord_list = chord_list
    def airfoil_interpolate(self,airfoil_df, chord):
        #airfoil = 'DAE31.dat'
        #airfoil_df = pd.read_table(airfoil, sep='\t', skiprows=1, header=None, engine='python')
        #print(airfoil_df)
        if type(airfoil_df.iloc[0, 0]) == str:
            airfoil_df_drop = airfoil_df.drop(airfoil_df.index[0])
        else:
            airfoil_df_drop = airfoil_df
        airfoil_df_float = airfoil_df_drop.astype('float64')
        airfoil_df_float_reset = airfoil_df_float.reset_index(drop=True)
        # airfoil_d_df = airfoil_df[airfoil_df[0] < self.max_airfoil_thickness_at[x] + 0.1].reset_index(drop=True)
        # border = airfoil_df_float[airfoil_df_float[1]>=0].min()
        border_idx = airfoil_df_float_reset[0].idxmin()
        border = airfoil_df_float_reset[0].min()
        if airfoil_df_float_reset[airfoil_df_float_reset[0] == border].shape[0] != 1:
            airfoil_df_float_min = airfoil_df_float_reset[airfoil_df_float_reset[0] == border]
            border_idx = airfoil_df_float_min[1].idxmin()

        airfoil_top_df, airfoil_bottom_df = airfoil_df_float_reset[:border_idx], airfoil_df_float_reset[border_idx:]
        # print(airfoil_top_df)
        airfoil_x_top_df, airfoil_y_top_df, airfoil_x_bottom_df, airfoil_y_bottom_df = airfoil_top_df[0], \
            airfoil_top_df[1], airfoil_bottom_df[0], airfoil_bottom_df[1]
        airfoil_x_top, airfoil_y_top, airfoil_x_bottom, airfoil_y_bottom = airfoil_x_top_df.to_numpy(), airfoil_y_top_df.to_numpy(), airfoil_x_bottom_df.to_numpy(), airfoil_y_bottom_df.to_numpy()
        airfoil_x_top_mm, airfoil_y_top_mm, airfoil_x_bottom_mm, airfoil_y_bottom_mm = airfoil_x_top * chord, airfoil_y_top * chord, airfoil_x_bottom * chord, airfoil_y_bottom * chord
        #print(airfoil_x_top_mm)
        airfoil_top_fx, airfoil_bottom_fx = interpolate.interp1d(airfoil_x_top_mm, airfoil_y_top_mm,
                                                              kind='cubic'), interpolate.interp1d(airfoil_x_bottom_mm,
                                                                                                   airfoil_y_bottom_mm,
                                                                                                   kind='cubic')
        # x_up, x_down = np.linspace(0, self.chord, self.airfoil_number_partition), np.linspace(0, self.chord,self.airfoil_number_partition)
        # y_up = airfoil_up_fx(x_up)
        # y_down = airfoil_down_fx(x_down)
        # fig, ax = plt.subplots(figsize=(40, 5))
        # plt.plot(x_up, airfoil_up_fx(x_up))
        return airfoil_top_fx, airfoil_bottom_fx
    def batch_interpolate(self):
        top_fx_list = []
        bottom_fx_list = []
        for airfoil_df, chord in zip(self.airfoil_df_list, self.chord_list):
            top_fx, bottom_fx = self.airfoil_interpolate(airfoil_df, chord)
            top_fx_list.append(top_fx)
            bottom_fx_list.append(bottom_fx)
        return [top_fx_list, bottom_fx_list]


if __name__ == '__main__':
    x = AirfoilInterpolate(800,100)
    x.airfoil_interpolate()
