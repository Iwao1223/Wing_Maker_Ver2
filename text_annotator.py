import numpy as np
import matplotlib.pyplot as plt
from geometry import Geometry

class TextAnnotator:
    def __init__(self, config):
        self.config = config
        self.zero_x = np.array([0, 0.1, 0.4, 0.5, 0.5, 0.4, 0.1, 0.05, 0.45, 0.05, 0, 0])
        self.one_x = np.array([0.25, 0.5, 0.5])
        self.two_x = np.array([0, 0.4, 0.5, 0.5, 0.4, 0.1, 0, 0, 0.5])
        self.three_x = np.array([0, 0.4, 0.5, 0.5, 0.4, 0.1, 0.4, 0.5, 0.5, 0.4, 0])
        self.four_x = np.array([0.15, 0, 0.5, 0.4, 0.4, 0.4])
        self.five_x = np.array([0.5, 0, 0, 0.4, 0.5, 0.5, 0.4, 0])
        self.six_x = np.array([0.5, 0.4, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.4, 0])
        self.seven_x = np.array([0, 0, 0.5, 0.35])
        self.eight_x = np.array([0, 0.1, 0.4, 0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.1, 0, 0, 0.1, 0.4, 0.1, 0, 0])
        self.nine_x = np.array([0.4, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.3])

        self.zero_y = np.array([0.9, 1, 1, 0.9, 0.1, 0, 0, 0.05, 0.95, 0.05, 0.1, 0.9])
        self.one_y = np.array([0.75, 1, 0])
        self.two_y = np.array([1, 1, 0.9, 0.6, 0.5, 0.5, 0.4, 0, 0])
        self.three_y = np.array([1, 1, 0.9, 0.6, 0.5, 0.5, 0.5, 0.4, 0.1, 0, 0])
        self.four_y = np.array([1, 0.5, 0.5, 0.5, 1, 0])
        self.five_y = np.array([1, 1, 0.5, 0.5, 0.4, 0.1, 0, 0])
        self.six_y = np.array([0.9, 1, 1, 0.9, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5])
        self.seven_y = np.array([0.75, 1, 1, 0])
        self.eight_y = np.array([0.9, 1, 1, 0.9, 0.6, 0.5, 0.4, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.5, 0.6, 0.9])
        self.nine_y = np.array([0.6, 0.6, 0.7, 0.9, 1, 1, 0.9, 0.8, 0])
        self.text_number_x = [self.zero_x, self.one_x, self.two_x, self.three_x, self.four_x, self.five_x, self.six_x,
                              self.seven_x,
                              self.eight_x, self.nine_x]
        self.text_number_y = [self.zero_y, self.one_y, self.two_y, self.three_y, self.four_y, self.five_y, self.six_y,
                              self.seven_y,
                              self.eight_y, self.nine_y]
        self.text_c_x = np.array([0.75, 0.25, 0, 0, 0.25, 0.75])
        self.text_c_y = np.array([1, 1, 0.75, 0.25, 0, 0])
        self.text_w_x = np.linspace(0, 1, 5)
        self.text_w_y = np.array([1, 0, 1, 0, 1])
        self.text_i_x = np.array([0.2, 0.8, 0.5, 0.5, 0.8, 0.2])
        self.text_i_y = np.array([0, 0, 0, 1, 1, 1])
        self.text_m_x = np.array([0.1, 0.1, 0.5, 0.9, 0.9])
        self.text_m_y = np.array([0, 1, 0.2, 1, 0])
        self.text_o_x = np.array([0.75, 0.25, 0, 0, 0.25, 0.75, 1, 1, 0.75])
        self.text_o_y = np.array([1, 1, 0.75, 0.25, 0, 0, 0.25, 0.75, 1])
        self.text_e_x = np.array([0.5, 0, 0, 0.5, 0, 0, 0.5])
        self.text_e_y = np.array([1, 1, 0.5, 0.5, 0.5, 0, 0])
        self.text_l_x = np.array([0.25, 0.25, 0.75])
        self.text_l_y = np.array([1.05, 0.3, 0.3])
        self.text_r_x = np.array([0.25, 0.25, 0.65, 0.75, 0.75, 0.65, 0.25, 0.45, 0.75]) * 1.2 - 0.15
        self.text_r_y = np.array([0.4, 1, 1, 0.9, 0.8, 0.7, 0.7, 0.7, 0.4]) * 1.2 - 0.2
        self.text_c_x = np.array([0.75, 0.25, 0, 0, 0.25, 0.75])
        self.text_c_y = np.array([1, 1, 0.75, 0.25, 0, 0])
        self.text_w_x = np.linspace(0, 1, 5)
        self.text_w_y = np.array([1, 0, 1, 0, 1])
        self.text_i_x = np.array([0.2, 0.8, 0.5, 0.5, 0.8, 0.2])
        self.text_i_y = np.array([0, 0, 0, 1, 1, 1])
        self.text_m_x = np.array([0.1, 0.1, 0.5, 0.9, 0.9])
        self.text_m_y = np.array([0, 1, 0.2, 1, 0])
        self.text_o_x = np.array([0.75, 0.25, 0, 0, 0.25, 0.75, 1, 1, 0.75])
        self.text_o_y = np.array([1, 1, 0.75, 0.25, 0, 0, 0.25, 0.75, 1])
        self.text_e_x = np.array([0.5, 0, 0, 0.5, 0, 0, 0.5])
        self.text_e_y = np.array([1, 1, 0.5, 0.5, 0.5, 0, 0])
        self.text_l_x = np.array([0.25, 0.25, 0.75])
        self.text_l_y = np.array([1, 0, 0])
        self.text_r_x = np.array([0.25, 0.25, 0.65, 0.75, 0.75, 0.65, 0.25, 0.45, 0.75])
        self.text_r_y = np.array([0.4, 1, 1, 0.9, 0.8, 0.7, 0.7, 0.7, 0.4])
        self.text_r_y = np.array([0, 0.6, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3, 0]) * 1.7
        self.text_dash_x = np.array([0.25, 0.75])
        self.text_dash_y = np.array([0.5, 0.5])
        self.text_wing_x = [self.text_c_x, self.text_i_x, self.text_m_x, self.text_o_x, self.text_e_x]
        self.text_wing_y = [self.text_c_y, self.text_i_y, self.text_m_y, self.text_o_y, self.text_e_y]
        self.text_l_r_x = [self.text_l_x, self.text_r_x]
        self.text_l_r_y = [self.text_l_y, self.text_r_y]

    def draw_text_horizon(self, l=False, c=1, number=19, coordinates=[0, 0], width=12, height=4):
        plt.figure(figsize=(width, height))

        scale_factor_h = height / 1  # 元のfigsize高さが12なのでそれを基準にスケール
        scale_factor_w = width / 6  # 元のfigsize幅が4なのでそれを基準にスケール
        scale_factor = min(scale_factor_h, scale_factor_w)  # 元のfigsize高さが12なのでそれを基準にスケール

        offset = 1 * scale_factor
        offset_small = 0.75 * scale_factor
        offset_large = 1.5 * scale_factor
        x_base, y_base = coordinates

        if l:
            l_r_x = self.text_l_x * scale_factor + x_base
            l_r_y = self.text_l_y * scale_factor + y_base
            plt.plot(l_r_x, l_r_y, label='L', color='red')
            l_r_tuple = Geometry.to_tuple_list(l_r_x, l_r_y)
        else:
            l_r_x = self.text_r_x * scale_factor + x_base
            l_r_y = self.text_r_y * scale_factor + y_base
            plt.plot(l_r_x, l_r_y, label='R', color='green')
            l_r_tuple = Geometry.to_tuple_list(l_r_x, l_r_y)

        x_base += offset_small

        dash1_x = self.text_dash_x * scale_factor + x_base
        dash1_y = self.text_dash_y * scale_factor + y_base
        dash1_tuple = Geometry.to_tuple_list(dash1_x, dash1_y)
        plt.plot(dash1_x, dash1_y, label='-', color='black')

        x_base += offset_small
        wing_name_x = self.text_wing_x[c] * scale_factor + x_base
        wing_name_y = self.text_wing_y[c] * scale_factor + y_base
        wing_name_tuple = Geometry.to_tuple_list(wing_name_x, wing_name_y)
        plt.plot(wing_name_x, wing_name_y, label='C', color='blue')

        x_base += offset
        w_x = self.text_w_x * scale_factor + x_base
        w_y = self.text_w_y * scale_factor + y_base
        w_tuple = Geometry.to_tuple_list(w_x, w_y)
        plt.plot(w_x, w_y, label='W', color='orange')
        x_base += offset

        dash2_x = self.text_dash_x * scale_factor + x_base
        dash2_y = self.text_dash_y * scale_factor + y_base
        dash2_tuple = Geometry.to_tuple_list(dash2_x, dash2_y)
        plt.plot(dash2_x, dash2_y, label='-', color='black')
        x_base += offset_small

        number0_tuple = None
        number1_tuple = None

        if number is not None and 0 <= number <= 9:
            number0_x = self.text_number_x[0] * scale_factor + x_base
            number0_y = self.text_number_y[0] * scale_factor + y_base
            number0_tuple = Geometry.to_tuple_list(number0_x, number0_y)
            plt.plot(number0_x, number0_y, label='0', color='black')
            x_base += offset

            number1_x = self.text_number_x[number] * scale_factor + x_base
            number1_y = self.text_number_y[number] * scale_factor + y_base
            number1_tuple = Geometry.to_tuple_list(number1_x, number1_y)
            plt.plot(number1_x, number1_y, label=str(number), color='black')
            x_base += offset
        elif number >= 10:
            tens = number // 10
            units = number % 10
            number0_x = self.text_number_x[tens] * scale_factor + x_base
            number0_y = self.text_number_y[tens] * scale_factor + y_base
            number0_tuple = Geometry.to_tuple_list(number0_x, number0_y)
            x_base += offset
            number1_x = self.text_number_x[units] * scale_factor + x_base
            number1_y = self.text_number_y[units] * scale_factor + y_base
            number1_tuple = Geometry.to_tuple_list(number1_x, number1_y)
            plt.plot(number0_x, number0_y, label=str(tens), color='black')
            x_base += offset
            plt.plot(number1_x, number1_y, label=str(units), color='black')
            x_base += offset
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        text_tuple_list = [l_r_tuple, dash1_tuple, wing_name_tuple, w_tuple, dash2_tuple, number0_tuple, number1_tuple]
        return text_tuple_list

    def draw_text_vertical(self, l=False, c=1, number=19, coordinates=[0, 0], width=4, height=12):
        plt.figure(figsize=(width, height))

        # 与えられた高さに基づいてスケールを計算
        scale_factor_h = height / 3  # 元のfigsize高さが12なのでそれを基準にスケール
        scale_factor_w = width / 2  # 元のfigsize幅が4なのでそれを基準にスケール
        scale_factor = min(scale_factor_h,scale_factor_w)  # 元のfigsize高さが12なのでそれを基準にスケール

        offset = 2.0 * scale_factor
        x_base, y_base = coordinates
        x_base += 0.3 * scale_factor
        l_r_scale = 1.5 * scale_factor

        # RまたはL
        if l:
            l_r_x = self.text_l_x * l_r_scale + x_base
            l_r_y = self.text_l_y * scale_factor + y_base
            plt.plot(l_r_x, l_r_y, label='L', color='red')
            l_r_tuple = Geometry.to_tuple_list(l_r_x, l_r_y)
        else:
            l_r_x = self.text_r_x * l_r_scale + x_base
            l_r_y = self.text_r_y * scale_factor + y_base
            plt.plot(l_r_x, l_r_y, label='R', color='green')
            l_r_tuple = Geometry.to_tuple_list(l_r_x, l_r_y)
        x_base -= 0.3 * scale_factor

        # CWを直後に配置（改行なし）
        y_base -= offset * 0.55
        c_x = self.text_wing_x[c] * scale_factor + x_base
        c_y = self.text_wing_y[c] * scale_factor + y_base
        plt.plot(c_x, c_y, label='C', color='blue')
        c_tuple = Geometry.to_tuple_list(c_x, c_y)
        x_base += offset * 0.5
        w_x = self.text_w_x * scale_factor + x_base
        w_y = self.text_w_y * scale_factor + y_base
        plt.plot(w_x, w_y, label='W', color='orange')
        w_tuple = Geometry.to_tuple_list(w_x, w_y)
        x_base -= offset * 0.5

        # 改行を入れて数字（2桁の場合も横並びで表示）
        y_base -= offset * 0.55
        number_tuple = None

        if number is not None and 0 <= number <= 9:
            number0_x = self.text_number_x[0] * scale_factor + x_base
            number0_y = self.text_number_y[0] * scale_factor + y_base
            number0_tuple = Geometry.to_tuple_list(number0_x, number0_y)
            plt.plot(number0_x, number0_y, label='0', color='black')
            number_x = self.text_number_x[number] * scale_factor + x_base + offset * 0.5
            number_y = self.text_number_y[number] * scale_factor + y_base
            plt.plot(number_x, number_y, label=str(number), color='black')
            number_tuple = Geometry.to_tuple_list(number_x, number_y)
            number_tuple = [number0_tuple, number_tuple]
        elif number >= 10:
            tens = number // 10
            units = number % 10

            # 十の位と一の位を横に並べて配置
            number_tens_x = self.text_number_x[tens] * scale_factor + x_base
            number_tens_y = self.text_number_y[tens] * scale_factor + y_base
            plt.plot(number_tens_x, number_tens_y, label=str(tens), color='black')
            number_tens_tuple = Geometry.to_tuple_list(number_tens_x, number_tens_y)

            # 一の位（横に配置）
            number_units_x = self.text_number_x[units] * scale_factor + x_base + offset * 0.5
            number_units_y = self.text_number_y[units] * scale_factor + y_base
            plt.plot(number_units_x, number_units_y, label=str(units), color='black')
            number_units_tuple = Geometry.to_tuple_list(number_units_x, number_units_y)
            number_tuple = [number_tens_tuple, number_units_tuple]

        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        return [l_r_tuple, c_tuple, w_tuple, number_tuple[0],number_tuple[1]]