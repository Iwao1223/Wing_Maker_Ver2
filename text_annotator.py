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
        self.text_wing_x = [self.text_c_x, self.text_i_x, self.text_m_x, self.text_o_x, self.text_e_x]
        self.text_wing_y = [self.text_c_y, self.text_i_y, self.text_m_y, self.text_o_y, self.text_e_y]
        self.text_l_r_x = [self.text_l_x, self.text_r_x]
        self.text_l_r_y = [self.text_l_y, self.text_r_y]

    def draw_text(self, l=False, c=1, number=9, coordinates=[0,0]):
        plt.figure(figsize=(12, 4))
        offset = 1.0
        x_base, y_base = coordinates

        if l:
            l_r_x = self.text_l_x + x_base
            l_r_y = self.text_l_y + y_base
            plt.plot(l_r_x, l_r_y, label='L', color='red')
            l_r_tuple = Geometry.to_tuple_list(l_r_x, l_r_y)
        else:
            l_r_x = self.text_r_x + x_base
            l_r_y = self.text_r_y + y_base
            plt.plot(l_r_x, l_r_y, label='R', color='green')
            l_r_tuple = Geometry.to_tuple_list(l_r_x, l_r_y)

        x_base += offset
        wing_name_x = self.text_wing_x[c] + x_base
        wing_name_y = self.text_wing_y[c] + y_base
        wing_name_tuple = Geometry.to_tuple_list(wing_name_x, wing_name_y)
        plt.plot(wing_name_x, wing_name_y, label='C', color='blue')

        x_base += offset
        w_x = self.text_w_x + x_base
        w_y = self.text_w_y + y_base
        w_tuple = Geometry.to_tuple_list(w_x, w_y)
        plt.plot(w_x, w_y, label='W', color='orange')
        x_base += offset

        if number is not None and 0 <= number <= 9:
            number0_x = self.text_number_x[0] + x_base
            number0_y = self.text_number_y[0] + y_base
            plt.plot(number0_x, number0_y, label='0', color='black')
            x_base += offset

            number1_x = self.text_number_x[number] + x_base
            number1_y = self.text_number_y[number] + y_base
            plt.plot(number1_x, number1_y, label=str(number), color='black')
            x_base += offset
        elif number >= 10:
            tens = number // 10
            units = number % 10
            plt.plot(self.text_number_x[tens] + x_base, self.text_number_y[tens], label=str(tens), color='black')
            x_base += offset
            plt.plot(self.text_number_x[units] + x_base, self.text_number_y[units], label=str(units), color='black')
            x_base += offset

        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
