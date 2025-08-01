# text_annotatorと、文字を入れたいPartオブジェクト（my_part）がある前提です
from text_annotator import TextAnnotator
from complete_nesting_algorithm_Version2_Version3 import Part
from dataclass import WingConfig
import numpy as np
from dxf_drawer import DXFDrawer
from dataclass import WingConfig
from wing_interpolate import WingInterpolate
from airfoil_calculator import AirfoilCalculator
from text_annotator import TextAnnotator
from dxf_drawer import DXFDrawer
from nesting_integrator import integrate_nesting_to_wingmaker
from shapely.geometry import MultiPoint


# configはご自身のプロジェクトに合わせて設定してください
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
text_annotator = TextAnnotator(config)
wing_interpolator = WingInterpolate(config)
calculator = AirfoilCalculator(config)
text_annotator = TextAnnotator(config)
drawer = DXFDrawer()

chord_list, mix_list, \
    spar_position_list, rear_spar_position_list, \
    spar_diameter_list, rear_spar_diameter_list, ellipse_degree_list = wing_interpolator.interpolate()
airfoil = calculator.generate_airfoil(chord_list, mix_list)
spar_position_tuple_list, rear_spar_position_tuple_list = calculator.get_spar_positions(spar_position_list,
                                                                                             rear_spar_position_list,
                                                                                             spar_diameter_list,
                                                                                             rear_spar_diameter_list)
spar_cap_tuple_list = calculator.generate_spar_cap(spar_diameter_list)
squid_cap_tuple_list = calculator.generate_squid_cap()
squid_cap_lightning_tuple_list = calculator.generate_squid_cap_lightning()
joint_top_tuple_list, joint_bottom_tuple_list = calculator.generate_joint()
squid_tuple_list = calculator.generate_squid()

# 仮のパーツを作成

my_part = Part("Sample_Part", joint_top_tuple_list[0])

# --- 前提 ---
# my_part と text_annotator は準備済みとします

print("テキストサイズの自動最適化を開始します...")

# --- 1. text_annotatorの「ズレ」の基本値を計測（これは初回のみ行います）---
CALIBRATION_WIDTH = 10.0
dummy_polylines_for_calib = text_annotator.draw_text_horizon(
    number=0, coordinates=[0, 0], width=CALIBRATION_WIDTH, height=3
)
all_points_calib = [p for poly in dummy_polylines_for_calib for p in poly]
if not all_points_calib:
    scale_factor_x = 1.0
    text_annotator_offset = np.array([0.0, 0.0])
else:
    min_x_calib = min(p[0] for p in all_points_calib)
    max_x_calib = max(p[0] for p in all_points_calib)
    min_y_calib = min(p[1] for p in all_points_calib)
    actual_width = max_x_calib - min_x_calib
    scale_factor_x = actual_width / CALIBRATION_WIDTH if CALIBRATION_WIDTH > 0 else 1.0
    text_annotator_offset = np.array([min_x_calib, min_y_calib])

print(
    f"--> 基本キャリブレーション完了 (Scale: {scale_factor_x:.4f}, Offset: [{text_annotator_offset[0]:.4f}, {text_annotator_offset[1]:.4f}])")

# --- 2. テキストがパーツに収まるまで、サイズを調整しながら試行錯誤します ---

# まずは大きめのサイズから試す
padding_ratio = 0.5
final_text_polylines = None
MAX_ITERATIONS = 15  # 無限ループを防ぐための最大試行回数
aspect_ratio = 5.0  # 文字の基本アスペクト比

part_bounds = my_part.polygon.bounds
part_width = part_bounds[2] - part_bounds[0]
part_height = part_bounds[3] - part_bounds[1]
center_point = my_part.polygon.representative_point()

for i in range(MAX_ITERATIONS):
    print(f"\n試行 {i + 1}/{MAX_ITERATIONS}: padding_ratio = {padding_ratio:.3f}")

    # a. 現在のpadding_ratioで、描画したい文字のサイズと位置を計算
    desired_text_width = part_width * padding_ratio
    desired_text_height = desired_text_width / aspect_ratio
    if desired_text_height > (part_height * padding_ratio):
        desired_text_height = part_height * padding_ratio
        desired_text_width = desired_text_height * aspect_ratio

    # b. キャリブレーション結果を使って、text_annotatorに渡す値を補正
    corrected_width = desired_text_width / scale_factor_x
    corrected_height = desired_text_height / scale_factor_x
    desired_coords = np.array([
        center_point.x - (desired_text_width / 2),
        center_point.y - (desired_text_height / 2)
    ])
    scaled_offset_x = text_annotator_offset[0] * (corrected_width / CALIBRATION_WIDTH)
    scaled_offset_y = text_annotator_offset[1]
    corrected_coords = desired_coords - np.array([scaled_offset_x, scaled_offset_y])

    # c. 補正値で実際にテキストを生成してみる
    generated_polylines = text_annotator.draw_text_horizon(
        number=1, coordinates=list(corrected_coords), width=corrected_width, height=corrected_height
    )

    # d. 生成された文字が、パーツの多角形の内側に完全に収まっているか判定
    if not generated_polylines or not any(generated_polylines):
        print(" -> テキストが生成されませんでした。スキップします。")
        padding_ratio *= 0.9  # サイズを小さくして再試行
        continue

    all_generated_points = [point for poly in generated_polylines for point in poly]

    # 生成された文字全体を囲む最小の四角形（エンベロープ）を作成
    generated_text_bbox = MultiPoint(all_generated_points).envelope

    # 判定！
    if generated_text_bbox.within(my_part.polygon):
        print(f" -> 成功！ このサイズでテキストが収まりました。")
        final_text_polylines = generated_polylines
        break  # 最適なサイズが見つかったのでループを抜ける
    else:
        print(f" -> はみ出しました。サイズを縮小して再試行します。")
        padding_ratio *= 0.95  # 少し小さくして次のループへ

    if i == MAX_ITERATIONS - 1:
        print("警告: 最大試行回数に達しました。最後に試したサイズを使用します。")
        final_text_polylines = generated_polylines

# --- 3. 最終的に決定したテキストを描画 ---
drawer.draw_polyline(list(my_part.polygon.exterior.coords))
if final_text_polylines:
    drawer.draw_text(final_text_polylines)

drawer.save('output_auto_fit.dxf')
print("\n処理完了。'output_auto_fit.dxf' を確認してください。")