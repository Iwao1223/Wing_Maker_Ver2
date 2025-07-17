# airfoil_calculator.py
from merge_airfoil import MergeAirfoil
from airfoil_interpolate import AirfoilInterpolate
from geometry import Geometry
import numpy as np
import matplotlib.pyplot as plt


class AirfoilCalculator:
    def __init__(self, config):
        self.config = config
        self.chord_list = None
        self.offset_fx_list_top = []
        self.offset_fx_list_bottom = []
        self.x_section_list_top = []
        self.x_section_list_bottom = []
        self.airfoil_tuple_list = []
        self.airfoil_x_list = []
        self.airfoil_y_list = []
        self.spar_position_tuple_list = []
        self.rear_spar_position_tuple_list = []
        self.spar_cap_tuple_list = []
        self.airfoil_fx_list = []
        self.joint_top_bottom_x_list = []
        self.joint_top_bottom_y_list = []
        self.joint_bottom_top_x_list = []
        self.joint_bottom_top_y_list = []
        self.squid_joint_x_list = []
        self.squid_joint_y_list = []

    def generate_airfoil(self, chord_list, mix_list):
        self.chord_list = chord_list
        airfoil = MergeAirfoil(self.config.airfoil_T, self.config.airfoil_R, mix_list)
        rib_cap_position = self.config.rib_cap_position * chord_list / 100
        plank_position_top = self.config.plank_position_top * chord_list / 100
        plank_position_bottom = self.config.plank_position_bottom * chord_list / 100
        airfoil_df_list = airfoil.merge()
        airfoil_fx = AirfoilInterpolate(airfoil_df_list, chord_list)
        self.airfoil_fx_list = airfoil_fx.batch_interpolate()
        plank_fx_list, rib_cap_fx_list, rib_cap_plank_fx_list, joint_fx_list = [], [], [], []
        for i in range(len(chord_list)):
            plank_fx = Geometry.offset_airfoil(self.airfoil_fx_list[0][i], self.airfoil_fx_list[1][i],
                                               self.config.plank_thickness)
            plank_fx_list.append(plank_fx)
            rib_cap_fx = Geometry.offset_airfoil(self.airfoil_fx_list[0][i], self.airfoil_fx_list[1][i],
                                                 self.config.rib_cap_thickness)
            rib_cap_fx_list.append(rib_cap_fx)
            rib_cap_plank_fx = Geometry.offset_airfoil(self.airfoil_fx_list[0][i], self.airfoil_fx_list[1][i],
                                                       (self.config.rib_cap_thickness + self.config.plank_thickness))
            rib_cap_plank_fx_list.append(rib_cap_plank_fx)
            joint_fx = Geometry.offset_airfoil(self.airfoil_fx_list[0][i], self.airfoil_fx_list[1][i], (
                    self.config.rib_cap_thickness + self.config.plank_thickness + self.config.joint_thickness))
            joint_fx_list.append(joint_fx)
            offset_fx_list_top = [plank_fx_list[i][0], rib_cap_plank_fx_list[i][0], joint_fx_list[i][0],
                                  rib_cap_fx_list[i][0]]
            offset_fx_list_bottom = [plank_fx_list[i][1], rib_cap_plank_fx_list[i][1], joint_fx_list[i][1],
                                     rib_cap_fx_list[i][1]]
            x_section_list_top = np.array(
                [plank_fx[0].x.min(), rib_cap_position[i], plank_position_top[i] - self.config.joint_width,
                 plank_position_top[i] + self.config.joint_width, chord_list[i] - self.config.squid_length])
            x_section_list_bottom = np.array(
                [plank_fx[1].x.min(), rib_cap_position[i], plank_position_bottom[i] - self.config.joint_width,
                 plank_position_bottom[i] + self.config.joint_width, chord_list[i] - self.config.squid_length])
            #print(x_section_list_top, x_section_list_bottom)

            """
            ここで、squidの位置を計算
            """

            if self.config.squid[i] and self.config.rib_cap_trailing[i]:
                # squidの位置を計算
                squid_x = [x_section_list_top[-1] - 10]
                squid_y = (offset_fx_list_top[-1](squid_x) + offset_fx_list_bottom[-1](squid_x)) / 2
            elif self.config.squid[i] and not self.config.rib_cap_trailing[i]:
                squid_x = [x_section_list_top[-1] - 10]
                squid_y = (offset_fx_list_top[0](squid_x) + offset_fx_list_bottom[0](squid_x)) / 2
            elif not self.config.squid[i] and self.config.rib_cap_trailing[i]:
                squid_x = []
                squid_y = []
                x_section_list_top[-1] = chord_list[i] - self.config.rib_cap_thickness
                x_section_list_bottom[-1] = chord_list[i] - self.config.rib_cap_thickness
            else:
                squid_x = []
                squid_y = []
                x_section_list_top[-1] = chord_list[i]
                x_section_list_bottom[-1] = chord_list[i]

            self.squid_joint_x_list.append(squid_x)
            self.squid_joint_y_list.append(squid_y)


            # ここで、aileronの位置を計算
            x_intersect_top = x_section_list_top[-1]
            x_intersect_bottom = x_section_list_bottom[-1]
            if self.config.aileron_position[i] == 0:
                aileron = False
                aileron_x_top = []
                aileron_x_bottom = []
                aileron_y_top = []
                aileron_y_bottom = []
            elif self.config.aileron_position[i] != 0 and self.config.rib_cap_trailing[i]:
                aileron = True
                aileron_x = self.config.aileron_position[i] * chord_list[i] / 100
                aileron_y = (offset_fx_list_top[-1](aileron_x) + offset_fx_list_bottom[-1](aileron_x)) / 2
                aileron_diameter = (offset_fx_list_top[-1](aileron_x) - offset_fx_list_bottom[-1](
                    aileron_x)) * self.config.aileron_diameter_coef
                x_intersect_top, y_intersect_top = Geometry.find_intersection_circle_curve(offset_fx_list_top[-1],
                                                                                           aileron_x, aileron_y,
                                                                                           aileron_diameter / 2,
                                                                                           upper=True)
                x_intersect_bottom, y_intersect_bottom = Geometry.find_intersection_circle_curve(
                    offset_fx_list_bottom[-1], aileron_x, aileron_y, aileron_diameter / 2, upper=False)
                aileron_x_top = np.linspace(aileron_x - aileron_diameter / 2, x_intersect_top, 100)
                aileron_x_bottom = np.linspace(aileron_x - aileron_diameter / 2, x_intersect_bottom, 100)
                aileron_y_top = Geometry.arc_y(aileron_x, aileron_y, aileron_diameter / 2, aileron_x_top, upper=True)
                aileron_y_bottom = Geometry.arc_y(aileron_x, aileron_y, aileron_diameter / 2, aileron_x_bottom,
                                                  upper=False)

            elif self.config.aileron_position[i] != 0 and not self.config.rib_cap_trailing[i]:
                aileron = True
                aileron_x = self.config.aileron_position[i] * chord_list[i] / 100
                aileron_y = (offset_fx_list_top[0](aileron_x) + offset_fx_list_bottom[0](aileron_x)) / 2
                aileron_diameter = (offset_fx_list_top[0](aileron_x) - offset_fx_list_bottom[0](
                    aileron_x)) * self.config.aileron_diameter_coef
                x_intersect_top, y_intersect_top = Geometry.find_intersection_circle_curve(offset_fx_list_top[0],
                                                                                           aileron_x, aileron_y,
                                                                                           aileron_diameter,
                                                                                           upper=True)
                x_intersect_bottom, y_intersect_bottom = Geometry.find_intersection_circle_curve(
                    offset_fx_list_bottom[0], aileron_x, aileron_y, aileron_diameter, upper=False)
                aileron_x_top = np.linspace(aileron_x - aileron_diameter / 2, x_intersect_top, 100)
                aileron_x_bottom = np.linspace(aileron_x - aileron_diameter / 2, x_intersect_bottom, 100)
                aileron_y_top = Geometry.arc_y(aileron_x, aileron_y, aileron_diameter / 2, aileron_x_top, upper=True)
                aileron_y_bottom = Geometry.arc_y(aileron_x, aileron_y, aileron_diameter / 2, aileron_x_bottom,
                                                  upper=False)

            """
            
            # ここで、rib_cap_leading, rib_cap_trailing, aileronの状態に応じて処理を分岐
            
            """

            if self.config.plank[i] and self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and \
                    aileron:
                # 1. 全てTrue
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
            elif self.config.plank[i] and self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and not \
                    aileron:
                # 2.aileronのみFalse
                pass
            elif self.config.plank[i] and self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[i] and \
                    aileron:
                # 3. rib_cap_trailingのみFalse
                offset_fx_list_top[-1] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[-1] = self.airfoil_fx_list[1][i]
                offset_fx_list_top[2] = rib_cap_fx_list[i][0]
                offset_fx_list_bottom[2] = rib_cap_fx_list[i][1]
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif self.config.plank[i] and self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[i] and not \
                    aileron:
                # 4. rib_cap_trailing, aileronがFalse
                offset_fx_list_top[2] = rib_cap_fx_list[i][0]
                offset_fx_list_bottom[2] = rib_cap_fx_list[i][1]
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif self.config.plank[i] and not self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and \
                    aileron:
                # 5. rib_cap_leadingのみFalse
                offset_fx_list_top[0:3] = [plank_fx_list[i][0], plank_fx_list[i][0], plank_fx_list[i][0]]
                offset_fx_list_bottom[0:3] = [plank_fx_list[i][1], plank_fx_list[i][1], plank_fx_list[i][1]]
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif self.config.plank[i] and not self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and not \
                    aileron:
                # 6. rib_cap_leading, aileronがFalse
                offset_fx_list_top[0:3] = [plank_fx_list[i][0], plank_fx_list[i][0], plank_fx_list[i][0]]
                offset_fx_list_bottom[0:3] = [plank_fx_list[i][1], plank_fx_list[i][1], plank_fx_list[i][1]]
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif self.config.plank[i] and not self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[i] and \
                    aileron:
                # 7. rib_cap_leading, rib_cap_trailingがFalse
                offset_fx_list_top[0:3] = [plank_fx_list[i][0], plank_fx_list[i][0], plank_fx_list[i][0]]
                offset_fx_list_bottom[0:3] = [plank_fx_list[i][1], plank_fx_list[i][1], plank_fx_list[i][1]]
                offset_fx_list_top[-1] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[-1] = self.airfoil_fx_list[1][i]
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif self.config.plank[i] and not self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[
                i] and not aileron:
                # 8. rib_cap_leading, rib_cap_trailing, aileronがFalse
                offset_fx_list_top[0:3] = [plank_fx_list[i][0], plank_fx_list[i][0], plank_fx_list[i][0]]
                offset_fx_list_bottom[0:3] = [plank_fx_list[i][1], plank_fx_list[i][1], plank_fx_list[i][1]]
                offset_fx_list_top[-1] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[-1] = self.airfoil_fx_list[1][i]
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif not self.config.plank[i] and self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and \
                    aileron:
                # 9. plankのみFalse
                offset_fx_list_top[0] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[0] = self.airfoil_fx_list[1][i]
                offset_fx_list_top[1:] = [rib_cap_fx_list[i][0], rib_cap_fx_list[i][0], rib_cap_fx_list[i][0]]
                offset_fx_list_bottom[1:] = [rib_cap_fx_list[i][1], rib_cap_fx_list[i][1], rib_cap_fx_list[i][1]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
            elif not self.config.plank[i] and self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and not \
                    aileron:
                # 10. plank, aileronがFalse
                offset_fx_list_top[0] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[0] = self.airfoil_fx_list[1][i]
                offset_fx_list_top[1:] = [rib_cap_fx_list[i][0], rib_cap_fx_list[i][0], rib_cap_fx_list[i][0]]
                offset_fx_list_bottom[1:] = [rib_cap_fx_list[i][1], rib_cap_fx_list[i][1], rib_cap_fx_list[i][1]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
            elif not self.config.plank[i] and self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[i] and \
                    aileron:
                # 11. plank, rib_cap_trailingがFalse
                offset_fx_list_top[0] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[0] = self.airfoil_fx_list[1][i]
                offset_fx_list_top[1:3] = [rib_cap_fx_list[i][0], rib_cap_fx_list[i][0]]
                offset_fx_list_bottom[1:3] = [rib_cap_fx_list[i][1], rib_cap_fx_list[i][1]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                offset_fx_list_top[-1] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[-1] = self.airfoil_fx_list[1][i]
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif not self.config.plank[i] and self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[
                i] and not aileron:
                # 12. plank, rib_cap_trailing, aileronがFalse
                offset_fx_list_top[0] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[0] = self.airfoil_fx_list[1][i]
                offset_fx_list_top[1:3] = [rib_cap_fx_list[i][0], rib_cap_fx_list[i][0]]
                offset_fx_list_bottom[1:3] = [rib_cap_fx_list[i][1], rib_cap_fx_list[i][1]]
                offset_fx_list_top[-1] = self.airfoil_fx_list[0][i]
                offset_fx_list_bottom[-1] = self.airfoil_fx_list[1][i]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif not self.config.plank[i] and not self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[i] and \
                    aileron:
                # 13. plank, rib_cap_leadingがFalse
                offset_fx_list_top[0:3] = [self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i]]
                offset_fx_list_bottom[0:3] = [self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif not self.config.plank[i] and not self.config.rib_cap_leading[i] and self.config.rib_cap_trailing[
                i] and not aileron:
                # 14. plank, rib_cap_leading, aileronがFalse
                offset_fx_list_top[0:3] = [self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i]]
                offset_fx_list_bottom[0:3] = [self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            elif not self.config.plank[i] and not self.config.rib_cap_leading[i] and not self.config.rib_cap_trailing[
                i] and aileron:
                # 15. plank, rib_cap_leading, rib_cap_trailingがFalse
                offset_fx_list_top[0:] = [self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i],
                                          self.airfoil_fx_list[0][i]]
                offset_fx_list_bottom[0:] = [self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i],
                                             self.airfoil_fx_list[1][i]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                x_section_list_top[-1] = x_intersect_top
                x_section_list_bottom[-1] = x_intersect_bottom
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]
            else:
                # 16. 全てFalse
                offset_fx_list_top[0:] = [self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i], self.airfoil_fx_list[0][i],
                                          self.airfoil_fx_list[0][i]]
                offset_fx_list_bottom[0:] = [self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i], self.airfoil_fx_list[1][i],
                                             self.airfoil_fx_list[1][i]]
                x_section_list_top[0] = self.airfoil_fx_list[0][i].x.min()
                x_section_list_bottom[0] = self.airfoil_fx_list[1][i].x.min()
                x_section_list_top[3] = plank_position_top[i]
                x_section_list_bottom[3] = plank_position_bottom[i]

            """
            #場合分け処理終了
            
            """
            if self.config.plank[i]:
                for j in np.linspace(4, x_section_list_top[0], 1000):
                    if offset_fx_list_top[0](j) > offset_fx_list_bottom[0](j):
                        x_section_list_top[0] = j
                        x_section_list_bottom[0] = j

            self.x_section_list_top.append(x_section_list_top)
            self.x_section_list_bottom.append(x_section_list_bottom)
            self.offset_fx_list_top.append(offset_fx_list_top)
            self.offset_fx_list_bottom.append(offset_fx_list_bottom)
            # セクションの区切り位置のリストを反転

            offset_fx_list_top = np.flip(offset_fx_list_top)
            x_section_list_top = np.flip(x_section_list_top)

            # セクションの区切り位置のリストからx座標を生成
            x_sections_top = [
                np.linspace(x_section_list_top[i], x_section_list_top[(i + 1)], 100)
                for i in range(len(x_section_list_top) - 1)
            ]

            x_sections_bottom = [
                np.linspace(x_section_list_bottom[i], x_section_list_bottom[(i + 1)], 100)
                for i in range(len(x_section_list_bottom) - 1)
            ]
            # ｘ座標のリストからy座標を生成
            y_sections_top = [offset_fx_list_top[j](x_sections_top[j]) for j in range(len(x_sections_top))]
            y_sections_bottom = [offset_fx_list_bottom[j](x_sections_bottom[j]) for j in range(len(x_sections_bottom))]

            if self.config.rib_cap_leading and self.config.rib_cap_trailing and self.config.plank:
                x_sections_top[1] = x_sections_top[1][10:-20]
                x_sections_bottom[2] = x_sections_bottom[2][10:-20]
                y_sections_top[1] = y_sections_top[1][10:-20]
                y_sections_bottom[2] = y_sections_bottom[2][10:-20]
                self.joint_top_bottom_x_list.append(np.flip(x_sections_top[1]))
                self.joint_top_bottom_y_list.append(np.flip(y_sections_top[1]))
                self.joint_bottom_top_x_list.append(np.flip(x_sections_bottom[2]))
                self.joint_bottom_top_y_list.append(np.flip(y_sections_bottom[2]))



            #すべての座標を結合してエアフォイルの座標を生成
            airfoil_x = np.concatenate([
                np.array(aileron_x_top),
                np.array(squid_x),
                *x_sections_top,
                *x_sections_bottom,
                np.flip(np.array(aileron_x_bottom))
            ])
            airfoil_y = np.concatenate([
                np.array(aileron_y_top),
                np.array(squid_y),
                *y_sections_top,
                *y_sections_bottom,
                np.flip(np.array(aileron_y_bottom))
            ])
            # 座標をタプルのリストに変換
            airfoil_tuple = Geometry.to_tuple_list(airfoil_x, airfoil_y)
            self.airfoil_x_list.append(airfoil_x)
            self.airfoil_y_list.append(airfoil_y)
            self.airfoil_tuple_list.append(airfoil_tuple)
        return self.airfoil_tuple_list

    def get_spar_positions(self, spar_position, rear_spar_position, spar_diameter_list, rear_spar_diameter_list):
        """
        # self.airfoil_x_listなどを使って桁座標計算
        """
        for i in range(len(self.chord_list)):
            spar_position_x = spar_position[i] * self.chord_list[i] / 100
            rear_spar_position_x = rear_spar_position[i] * self.chord_list[i] / 100
            x_section_list_top = self.x_section_list_top[i]
            x_section_list_bottom = self.x_section_list_bottom[i]
            offset_fx_list_top = self.offset_fx_list_top[i]
            offset_fx_list_bottom = self.offset_fx_list_bottom[i]
            spar_section_top_x = Geometry.find_section_index(spar_position_x, x_section_list_top)
            rear_spar_section_top_x = Geometry.find_section_index(rear_spar_position_x, x_section_list_top)
            spar_section_bottom_x = Geometry.find_section_index(spar_position_x, x_section_list_bottom)
            rear_spar_section_bottom_x = Geometry.find_section_index(rear_spar_position_x, x_section_list_bottom)
            spar_position_y = (offset_fx_list_top[spar_section_top_x](spar_position_x) +
                               offset_fx_list_bottom[spar_section_bottom_x](spar_position_x)) / 2
            rear_spar_position_y = (offset_fx_list_top[rear_spar_section_top_x](rear_spar_position_x) +
                                    offset_fx_list_bottom[rear_spar_section_bottom_x](rear_spar_position_x)) / 2
            if (offset_fx_list_top[spar_section_top_x](spar_position_x) -
offset_fx_list_bottom[spar_section_bottom_x](spar_position_x)) < spar_diameter_list[i][0]:
                print('警告:{}番目のリブが桁がはみ出します'.format(i + 1))
            if (offset_fx_list_top[rear_spar_section_top_x](rear_spar_position_x) -
                offset_fx_list_bottom[rear_spar_section_bottom_x](rear_spar_position_x)) < rear_spar_diameter_list[i]:
                print('警告:{}番目のリブがリアスパ桁がはみ出します'.format(i + 1))
            # spar_position_tupleとrear_spar_position_tupleを生成
            spar_position_tuple = (spar_position_x, spar_position_y)
            rear_spar_position_tuple = (rear_spar_position_x, rear_spar_position_y)
            self.spar_position_tuple_list.append(spar_position_tuple)
            self.rear_spar_position_tuple_list.append(rear_spar_position_tuple)

        return self.spar_position_tuple_list, self.rear_spar_position_tuple_list

    def generate_spar_cap(self, spar_diameter_list):
        for i in range(len(self.chord_list)):
            spar_cap_top = []
            spar_cap_bottom = []
            spar_position_x, spar_position_y = self.spar_position_tuple_list[i]
            spar_diameter = spar_diameter_list[i]
            airfoil_fx_top = self.offset_fx_list_top[i][0]
            airfoil_fx_bottom = self.airfoil_fx_list[1][i]
            spar_cap_x_leading = spar_position_x - spar_diameter[0] / 2 - self.config.spar_cap_thickness
            spar_cap_x_trailing = spar_position_x + spar_diameter[0] / 2 + self.config.spar_cap_thickness
            spar_cap_height_leading = airfoil_fx_top(spar_cap_x_leading) - spar_position_y
            spar_cap_height_trailing = airfoil_fx_bottom(spar_cap_x_trailing) - spar_position_y
            spar_cap_x_top_leading = spar_cap_x_leading + (spar_cap_height_leading / np.tan(np.deg2rad(self.config.spar_cap_degree)))
            spar_cap_x_top_trailing = spar_cap_x_trailing + (spar_cap_height_trailing / np.tan(np.deg2rad(self.config.spar_cap_degree)))
            spar_cap_x_section = np.linspace(spar_cap_x_top_leading, spar_cap_x_top_trailing, 100)
            spar_cap_y_section_top = airfoil_fx_top(spar_cap_x_section)
            spar_cap_y_section_bottom = airfoil_fx_bottom(spar_cap_x_section)
            spar_cap_x = np.concatenate((
                np.array([spar_cap_x_leading]),
                spar_cap_x_section,
                np.array([spar_cap_x_trailing]),
                spar_cap_x_section[::-1]
            ))
            spar_cap_y = np.concatenate((
                np.array([spar_position_y]),
                spar_cap_y_section_top,
                np.array([spar_position_y]),
                spar_cap_y_section_bottom[::-1]
            ))
            spar_cap_tuple = Geometry.to_tuple_list(spar_cap_x, spar_cap_y)
            self.spar_cap_tuple_list.append(spar_cap_tuple)
        return self.spar_cap_tuple_list

    def generate_joint(self):
        joint_top_tuple_list = []
        joint_bottom_tuple_list = []
        if self.config.rib_cap_leading and self.config.rib_cap_trailing and self.config.plank:
            for i in range(len(self.chord_list)):
                plank_position_top = self.config.plank_position_top * self.chord_list[i] / 100
                plank_position_bottom = self.config.plank_position_bottom * self.chord_list[i] / 100
                joint_x_top_top_leading = np.linspace(self.x_section_list_top[i][2], plank_position_top, 100)
                joint_x_top_top_trailing = np.linspace(plank_position_top, self.x_section_list_top[i][3], 100)
                joint_x_bottom_bottom_leading = np.linspace(self.x_section_list_bottom[i][2], plank_position_bottom, 100)
                joint_x_bottom_bottom_trailing = np.linspace(plank_position_bottom, self.x_section_list_bottom[i][3], 100)
                joint_y_top_top_leading = self.offset_fx_list_top[i][1](joint_x_top_top_leading)
                joint_y_top_top_trailing = self.offset_fx_list_top[i][3](joint_x_top_top_trailing)
                joint_y_bottom_bottom_leading = self.offset_fx_list_bottom[i][1](joint_x_bottom_bottom_leading)
                joint_y_bottom_bottom_trailing = self.offset_fx_list_bottom[i][3](joint_x_bottom_bottom_trailing)
                joint_x_top = np.concatenate((
                    joint_x_top_top_leading,
                    joint_x_top_top_trailing,
                    self.joint_top_bottom_x_list[i][::-1]
                ))
                joint_y_top = np.concatenate((
                    joint_y_top_top_leading,
                    joint_y_top_top_trailing,
                    self.joint_top_bottom_y_list[i][::-1]
                ))
                joint_x_bottom = np.concatenate((
                    joint_x_bottom_bottom_leading,
                    joint_x_bottom_bottom_trailing,
                    self.joint_bottom_top_x_list[i]
                ))
                joint_y_bottom = np.concatenate((
                    joint_y_bottom_bottom_leading,
                    joint_y_bottom_bottom_trailing,
                    self.joint_bottom_top_y_list[i]
                ))
                joint_top_tuple = Geometry.to_tuple_list(joint_x_top, joint_y_top)
                joint_bottom_tuple = Geometry.to_tuple_list(joint_x_bottom, joint_y_bottom)
                joint_top_tuple_list.append(joint_top_tuple)
                joint_bottom_tuple_list.append(joint_bottom_tuple)
        else:
            print('警告: リブキャップリーディング、リブキャップトレーリング、プランクのいずれかがFalseの場合、ジョイントは生成されません。')
        return joint_top_tuple_list, joint_bottom_tuple_list

    def generate_squid(self):
        squid_tuple_list = []
        if self.config.squid is None:
            print('警告: squidの設定がありません。')
            return squid_tuple_list
        geso_length = self.config.squid_length/ 3
        for i in range(len(self.chord_list)):
            te_carbon_fx = Geometry.offset_airfoil(self.airfoil_fx_list[0][i], self.airfoil_fx_list[1][i], self.config.te_carbon_thickness)
            squid_x_geso = np.linspace(self.chord_list[i] - self.config.squid_length, self.chord_list[i] - self.config.squid_length + geso_length, 100)
            squid_x_atama = np.linspace(self.chord_list[i] - self.config.squid_length + geso_length, self.chord_list[i] - self.config.te_length, 100)
            squid_x_atama_carbon = np.linspace(self.chord_list[i] - self.config.te_length, self.chord_list[i] - self.config.te_carbon_thickness, 100)
            squid_y_geso_top = self.offset_fx_list_top[i][-1](squid_x_geso)
            squid_y_atama_top = self.airfoil_fx_list[0][i](squid_x_atama)
            squid_y_atama_carbon_top = te_carbon_fx[0](squid_x_atama_carbon)
            squid_y_geso_bottom = self.offset_fx_list_bottom[i][-1](squid_x_geso)
            squid_y_atama_bottom = self.airfoil_fx_list[1][i](squid_x_atama)
            squid_y_atama_carbon_bottom = te_carbon_fx[1](squid_x_atama_carbon)
            squid_x = np.concatenate((self.squid_joint_x_list[i],squid_x_geso, squid_x_atama, squid_x_atama_carbon,
                                      np.flip(squid_x_atama_carbon), np.flip(squid_x_atama), np.flip(squid_x_geso)))
            squid_y = np.concatenate((self.squid_joint_y_list[i],squid_y_geso_top, squid_y_atama_top, squid_y_atama_carbon_top,
                                          np.flip(squid_y_atama_carbon_bottom), np.flip(squid_y_atama_bottom),
                                          np.flip(squid_y_geso_bottom)))

            squid_tuple = Geometry.to_tuple_list(squid_x, squid_y)
            squid_tuple_list.append(squid_tuple)
            squid_tuple_list.append([])

        return squid_tuple_list

    def get_rib_cap_length(self):
        rib_cap_length = []
        geso_length = self.config.squid_length / 3
        if not self.config.rib_cap_leading and not self.config.rib_cap_trailing:
            print('警告: リブキャップリーディング、リブキャップトレーリングの両方かがFalseの場合、リブキャップは生成されません。')
            return []
        for i in range(len(self.chord_list)):
            plank_position_top = self.config.plank_position_top * self.chord_list[i] / 100
            plank_position_bottom = self.config.plank_position_bottom * self.chord_list[i] / 100
            rib_cap_top_leading_x = np.linspace(self.x_section_list_top[i][1], plank_position_top, 100)
            rib_cap_bottom_leading_x = np.linspace(self.x_section_list_bottom[i][1], plank_position_bottom, 100)
            rib_cap_top_trailing_x = np.linspace(plank_position_top, self.chord_list[i] - self.config.squid_length + geso_length, 100)
            rib_cap_bottom_trailing_x = np.linspace(plank_position_bottom, self.chord_list[i] - self.config.squid_length + geso_length, 100)
            rib_cap_top_leading_y = self.offset_fx_list_top[i][1](rib_cap_top_leading_x)
            rib_cap_bottom_leading_y = self.offset_fx_list_bottom[i][1](rib_cap_bottom_leading_x)
            rib_cap_top_trailing_y = self.offset_fx_list_top[i][3](rib_cap_top_trailing_x)
            rib_cap_bottom_trailing_y = self.offset_fx_list_bottom[i][3](rib_cap_bottom_trailing_x)
            rib_cap_top_leading_length = Geometry.trigonometry(rib_cap_top_leading_x, rib_cap_top_leading_y)
            rib_cap_bottom_leading_length = Geometry.trigonometry(rib_cap_bottom_leading_x, rib_cap_bottom_leading_y)
            rib_cap_top_trailing_length = Geometry.trigonometry(rib_cap_top_trailing_x, rib_cap_top_trailing_y)
            rib_cap_bottom_trailing_length = Geometry.trigonometry(rib_cap_bottom_trailing_x, rib_cap_bottom_trailing_y)
            if not self.config.rib_cap_leading[i]:
                rib_cap_top_leading_length = []
                rib_cap_bottom_leading_length = []
            if not self.config.rib_cap_trailing[i]:
                rib_cap_top_trailing_length = []
                rib_cap_bottom_trailing_length = []
            rib_cap_length_list = [rib_cap_top_leading_length, rib_cap_top_trailing_length,
                                   rib_cap_bottom_leading_length, rib_cap_bottom_trailing_length]
            rib_cap_length.append(rib_cap_length_list)
        return rib_cap_length

