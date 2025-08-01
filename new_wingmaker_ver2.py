"""
ネスティング機能を追加したWing_Maker_Ver2
"""
# new_wingmaker.py
from dataclass import WingConfig
from wing_interpolate import WingInterpolate
from airfoil_calculator import AirfoilCalculator
from text_annotator import TextAnnotator
from dxf_drawer import DXFDrawer
from nesting_integrator import integrate_nesting_to_wingmaker
import numpy as np


class NewWingMakerWithNesting:
    def __init__(self, config: WingConfig):
        self.config = config
        self.wing_interpolator = WingInterpolate(config)
        self.calculator = AirfoilCalculator(config)
        self.text_annotator = TextAnnotator(config)
        self.drawer = DXFDrawer()

    def make_rib(self):
        """既存のリブ生成機能"""
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

        text = self.text_annotator.draw_text_horizon(False, 0, 1, [0, 0], 5, 5)
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

        return airfoil

    def make_nested_ribs(self, bin_width: float = 1000.0,
                         simplify_tolerance: float = 0.5,
                         output_filename: str = "nested_ribs.dxf",
                         # ▼▼▼ safety_margin引数を追加 ▼▼▼
                         safety_margin: float = 0.0,
                         generate_video=False,
                         video_filename="nesting_process.mp4"):
        """
        ネスティング機能付きリブ生成

        Args:
            bin_width: ネスティングビンの幅（mm）
            simplify_tolerance: 翼型近似の許容誤差
            output_filename: 出力DXFファイル名

        Returns:
            ネスティング結果の辞書
        """
        print("=== ネスティング機能付きリブ生成開始 ===")

        # 翼型データの生成
        chord_list, mix_list, \
            spar_position_list, rear_spar_position_list, \
            spar_diameter_list, rear_spar_diameter_list, ellipse_degree_list = self.wing_interpolator.interpolate()

        airfoil_tuple_lists = self.calculator.generate_airfoil(chord_list, mix_list)

        # ★ 桁の位置情報を取得
        spar_position_tuple_list, rear_spar_position_tuple_list = self.calculator.get_spar_positions(
            spar_position_list,
            rear_spar_position_list,
            spar_diameter_list,
            rear_spar_diameter_list
        )

        print(f"生成された翼型数: {len(airfoil_tuple_lists)}")



        # ネスティング実行
        nesting_result = integrate_nesting_to_wingmaker(
            airfoil_tuple_lists=airfoil_tuple_lists,
            bin_width=bin_width,
            simplify_tolerance=simplify_tolerance,
            output_filename=output_filename,
            safety_margin=safety_margin,
            generate_video=generate_video,
            video_filename=video_filename,
            spar_positions=spar_position_tuple_list,
            rear_spar_positions=rear_spar_position_tuple_list,
            spar_diameters=spar_diameter_list,
            rear_spar_diameters=rear_spar_diameter_list,
            ellipse_degrees=ellipse_degree_list,
            circle_or_ellipse=self.config.circle_or_ellipse
        )

        if nesting_result["success"]:
            print("\n=== ネスティング完了 ===")
            print(f"出力ファイル: {nesting_result['output_file']}")
            print(f"材料サイズ: {nesting_result['bin_width']:.0f} × {nesting_result['bin_height']:.0f} mm")
            print(f"材料利用率: {nesting_result['utilization']:.2%}")
            print(f"配置部品数: {nesting_result['parts_count']}")
        else:
            print(f"エラー: {nesting_result['error']}")

        return nesting_result


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
        squid_cap_text_wide=10,
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

    wing_maker = NewWingMakerWithNesting(config)

    # 通常のリブ生成
    # wing_maker.make_rib()

    # ネスティング機能付きリブ生成
    result = wing_maker.make_nested_ribs(
        bin_width=910.0,  # 材料幅1200mm
        simplify_tolerance=2,  # 2mm以下の変化を無視
        output_filename="optimized_wing_layout.dxf",
        safety_margin=1.0,
        generate_video=False,
        video_filename="wing_rib_optimization.mp4"
    )