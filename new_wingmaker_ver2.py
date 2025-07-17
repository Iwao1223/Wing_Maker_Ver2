# new_wingmaker.py
from dataclass import WingConfig
from wing_interpolate import WingInterpolate
from airfoil_calculator import AirfoilCalculator
from text_annotator import TextAnnotator
from dxf_drawer import DXFDrawer
import numpy as np



class NewWingMaker:
    def __init__(self, config: WingConfig):
        self.config = config
        self.wing_interpolator = WingInterpolate(config)
        self.calculator = AirfoilCalculator(config)
        self.text_annotator = TextAnnotator(config)
        self.drawer = DXFDrawer()

    def make_rib(self):
        chord_list, mix_list, \
            spar_position_list, rear_spar_position_list, \
            spar_diameter_list, rear_spar_diameter_list, ellipse_degree_list = self.wing_interpolator.interpolate()
        airfoil = self.calculator.generate_airfoil(chord_list, mix_list)
        spar_position_tuple_list, rear_spar_position_tuple_list = self.calculator.get_spar_positions(spar_position_list,
                                                               rear_spar_position_list,
                                                               spar_diameter_list,
                                                               rear_spar_diameter_list)
        spar_cap_tuple_list = self.calculator.generate_spar_cap(spar_diameter_list)
        squid_cap_tuple_list = self.calculator.generate_squid_cap()
        squid_cap_lightning_tuple_list = self.calculator.generate_squid_cap_lightning()
        joint_top_tuple_list, joint_bottom_tuple_list = self.calculator.generate_joint()
        squid_tuple_list = self.calculator.generate_squid()
        rib_cap_length = self.calculator.get_rib_cap_length()
        text = self.text_annotator.draw_text_horizon(False,0,1,[0,0],5,5)
        print(text)
        self.drawer.draw_text(text)

        text = self.text_annotator.draw_text_horizon()
        self.drawer.draw_polyline(airfoil[0])

        self.drawer.draw_polyline(spar_cap_tuple_list[0])
        if self.config.circle_or_ellipse == 'circle':
            self.drawer.draw_circle(spar_position_tuple_list[0], spar_diameter_list[0])
        elif self.config.circle_or_ellipse == 'ellipse':
            self.drawer.draw_ellipse(spar_position_tuple_list[0], spar_diameter_list[0], ellipse_degree_list[0])
        self.drawer.draw_polyline(squid_cap_tuple_list[0])
        self.drawer.draw_polyline(squid_cap_lightning_tuple_list[0])
        self.drawer.draw_circle(rear_spar_position_tuple_list[0], rear_spar_diameter_list[0])
        self.drawer.draw_polyline(joint_top_tuple_list[0])
        self.drawer.draw_polyline(joint_bottom_tuple_list[0])
        self.drawer.draw_polyline(squid_tuple_list[0])
        self.drawer.save('airfoil1.dxf')


if __name__ == '__main__':
    config = WingConfig(
        rib_station_list=np.array([0, 200, 400, 600, 800, 900, 1000, 1200, 1400]),
        strut_station_list=np.array([0, 1750]),
        aileron_position=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        brock_span=1400,
        chord_R=600,
        chord_T=450,
        airfoil_R='opt_airfoil(7)',
        airfoil_T='opt_airfoil(7)',
        spar_position_R=25,
        spar_position_T=33,
        spar_major_diameter_R=40,
        spar_major_diameter_T=40,
        spar_minor_diameter_R=30,
        spar_minor_diameter_T=30,
        ellipse_degree_R=0,
        ellipse_degree_T=70,
        rear_spar_position_R=83.35904498,
        rear_spar_position_T=83.27261482,
        rear_spar_diameter_R=8,
        rear_spar_diameter_T=8,
        aileron_diameter_coef=1.1,
        plank_thickness=3,
        rib_cap_thickness=1,
        plank_position_top=67.75336121,
        plank_position_bottom=12,
        rib_cap_position=3,
        spar_cap_thickness=10,
        spar_cap_degree=70,
        squid_cap_thickness=1,
        squid_cap_length=20,
        squid_cap_text_wide = 10,
        joint_thickness=4,
        squid_length=50,
        te_length=30,
        te_carbon_thickness=0.2,
        joint_width=20,
        rib_interval=200,
        rib_thickness=5,
        strut_thickness=8,
        x_bracing_pit_diameter=13,
        brock_name='L-CW_',
        brock_number=0,
        text_position=77.75336121,
        left_or_right='left',
        plank=[True, True, True, True, True, True, True, True, True],
        rib_cap_leading=[True, True, True, True, True, True, True, True, True],
        rib_cap_trailing=[True, True, True, True, True, True, True, True, True],
        spar_cap=[True, True, True, True, True, True, True, True, True],
        rear_spar_cap=[True, True, True, True, True, True, True, True, True],
        squid=[True, True, True, True, True, True, True, True, True],
        only_rib=[False, False, False, False, False, False, False, False, False],
        only_plank=[False, False, False, False, False, False, False, False, False],
        circle_or_ellipse='ellipse',  # 'circle' or 'ellipse'

    )
    wing_maker = NewWingMaker(config)
    wing_maker.make_rib()
