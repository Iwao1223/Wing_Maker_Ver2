"""
Wing_Maker_Ver2のネスティング統合モジュール
翼型部品の近似・配置・DXF出力を行う
"""
import math

import numpy as np
import random
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt
from typing import List, Tuple
from complete_nesting_algorithm_Version2_Version3 import (
    Part, NestingAlgorithm, ImprovedBinPacking, GeneticAlgorithm, NFPCalculator
)
from dxf_drawer import DXFDrawer
import os
import shutil
import imageio
import copy

class WingNestingIntegrator:
    """Wing_Maker_Ver2のネスティング統合クラス"""

    def __init__(self, airfoil_tuple_lists, bin_width: float = 1000.0, simplify_tolerance: float = 0.5,
                 safety_margin: float = 0.0, spar_positions=None, rear_spar_positions=None, spar_diameters=None,
                 rear_spar_diameters=None, ellipse_degrees=None, circle_or_ellipse='circle'):
        self.airfoil_tuple_lists = airfoil_tuple_lists
        self.bin_width = bin_width
        self.simplify_tolerance = simplify_tolerance
        self.safety_margin = safety_margin  # ★ マージンを保存
        self.drawer = DXFDrawer()
        self.nesting_algorithm = None
        # --- ▼▼▼ ここから追加 ▼▼▼ ---
        self.spar_positions = spar_positions if spar_positions is not None else []
        self.rear_spar_positions = rear_spar_positions if rear_spar_positions is not None else []
        self.spar_diameters = spar_diameters if spar_diameters is not None else []
        self.rear_spar_diameters = rear_spar_diameters if rear_spar_diameters is not None else []
        self.ellipse_degrees = ellipse_degrees if ellipse_degrees is not None else []
        self.circle_or_ellipse = circle_or_ellipse
        # --- ▲▲▲ ここまで追加 ▲▲▲ ---
        
    def simplify_airfoil_tuples(self, tuple_lists: List[List[Tuple[float, float]]], 
                               tolerance: float = None) -> List[List[Tuple[float, float]]]:
        """
        翼型のタプルリストを近似して頂点数を削減
        
        Args:
            tuple_lists: 翼型の座標タプルリストのリスト
            tolerance: 近似の許容誤差（デフォルトは初期化時の値）
        
        Returns:
            近似されたタプルリストのリスト
        """
        if tolerance is None:
            tolerance = self.simplify_tolerance
            
        simplified_lists = []
        
        for i, tuple_list in enumerate(tuple_lists):
            if not tuple_list:
                simplified_lists.append([])
                continue
                
            try:
                # ShapelyのPolygonに変換
                original_poly = Polygon(tuple_list)
                
                # 近似処理
                simplified_poly = original_poly.simplify(tolerance, preserve_topology=True)
                
                # タプルリストに戻す
                if simplified_poly.is_empty:
                    print(f"警告: 翼型{i}の近似結果が空になりました。元のデータを使用します。")
                    simplified_lists.append(tuple_list)
                else:
                    simplified_coords = list(simplified_poly.exterior.coords)[:-1]  # 最後の重複点を除去
                    simplified_lists.append(simplified_coords)
                    
                    # 近似効果をログ出力
                    original_points = len(tuple_list)
                    simplified_points = len(simplified_coords)
                    reduction_rate = (1 - simplified_points / original_points) * 100
                    print(f"翼型{i}: {original_points}点 → {simplified_points}点 "
                          f"({reduction_rate:.1f}%削減)")
                    # 近似後の頂点数を確認
                    print("近似後の頂点数:", len(list(simplified_poly.exterior.coords)))
                    # ポリゴンを描画する例
                    fig, ax = plt.subplots(figsize=(16.0, 2.0))
                    original_points = list(original_poly.exterior.coords)
                    points_to_draw = list(simplified_poly.exterior.coords)
                    x, y = zip(*points_to_draw)
                    ax.plot(*zip(*original_points), color='red', label='Original Airfoil', alpha=0.5)
                    ax.fill(x, y, alpha=0.8, color='blue')
                    plt.show()
                    
            except Exception as e:
                print(f"警告: 翼型{i}の近似中にエラーが発生しました: {e}")
                print("元のデータを使用します。")
                simplified_lists.append(tuple_list)
                
        return simplified_lists
    
    def create_parts_from_airfoils(self, airfoil_tuple_lists: List[List[Tuple[float, float]]], 
                                   part_name_prefix: str = "Rib") -> List[Part]:
        """
        翼型のタプルリストからPart オブジェクトを作成
        
        Args:
            airfoil_tuple_lists: 翼型の座標タプルリストのリスト
            part_name_prefix: 部品名のプレフィックス
            
        Returns:
            Partオブジェクトのリスト
        """
        parts = []
        
        for i, tuple_list in enumerate(airfoil_tuple_lists):
            if not tuple_list:
                continue
                
            part_id = f"{part_name_prefix}_{i+1:03d}"
            
            try:
                # 座標の妥当性チェック
                if len(tuple_list) < 3:
                    print(f"警告: {part_id}の座標点が不足しています。スキップします。")
                    continue
                    
                part = Part(part_id, tuple_list)
                
                # 部品の妥当性チェック
                if not part.polygon.is_valid:
                    print(f"警告: {part_id}のポリゴンが無効です。修復を試みます。")
                    part.polygon = part.polygon.buffer(0)  # 自己修復
                    
                if part.polygon.is_valid and part.get_area() > 1.0:
                    parts.append(part)
                    print(f"{part_id}を作成: 面積={part.get_area():.2f}")
                else:
                    print(f"警告: {part_id}は無効またはサイズが小さすぎるためスキップします。")
                    
            except Exception as e:
                print(f"エラー: {part_id}の作成中にエラーが発生しました: {e}")
                
        return parts
    
    def run_nesting(self)  -> tuple:
        """
        ネスティングを実行
        
        Args:
            airfoil_tuple_lists: 翼型の座標タプルリストのリスト
            
        Returns:
            配置結果のImprovedBinPackingオブジェクト
        """
        print("=== Wing_Maker_Ver2 ネスティング開始 ===")
        
        # 1. 翼型データの近似
        print(f"\nステップ1: 翼型データの近似（許容誤差: {self.simplify_tolerance}）")
        simplified_airfoils = self.simplify_airfoil_tuples(self.airfoil_tuple_lists)
        
        # 2. Part オブジェクトの作成
        print("\nステップ2: Part オブジェクトの作成")
        parts = self.create_parts_from_airfoils(simplified_airfoils)
        
        if not parts:
            print("エラー: 有効な部品が作成されませんでした。")
            return None
            
        print(f"作成された部品数: {len(parts)}")
        
        # 3. ネスティング実行
        print("\nステップ3: ネスティング実行")
        # アルゴリズムのインスタンスを self に保存
        self.nesting_algorithm = NestingAlgorithm(parts, self.bin_width, safety_margin=self.safety_margin)
        # 2つの戻り値（配置結果と解）を受け取る
        result_packer, final_solution = self.nesting_algorithm.run()

        # 3つの値を返す
        return result_packer, final_solution, parts

    def export_nesting_to_dxf(self, result_packer: ImprovedBinPacking,
                              filename: str = "nesting_result.dxf") -> None:
        """
        ネスティング結果をDXFファイルに出力（桁情報も含む）
        """
        if not result_packer or not result_packer.placed_parts:
            print("エラー: 出力する部品がありません。")
            return

        print(f"\nステップ4: DXF出力 ({filename})")

        self.drawer = DXFDrawer()

        for part_info in result_packer.placed_parts:
            try:
                part_index = int(part_info.id.split('_')[-1]) - 1

                # --- 1. リブの外形線の変換（前回と同じロジック） ---
                original_points = self.airfoil_tuple_lists[part_index]
                if not original_points or len(original_points) < 3:
                    continue

                original_poly = Polygon(original_points)
                rotated_poly = rotate(original_poly, part_info.rotation, origin='centroid')
                minx, miny, _, _ = rotated_poly.bounds
                target_position = part_info.position
                dx = target_position[0] - minx
                dy = target_position[1] - miny

                final_poly = translate(rotated_poly, xoff=dx, yoff=dy)
                final_coords = list(final_poly.exterior.coords)
                self.drawer.draw_polyline(final_coords)
                print(f"部品 {part_info.id} を描画: 位置({target_position[0]:.1f}, {target_position[1]:.1f})")

                # --- 2. 桁の中心座標を同じように変換して描画 ---
                rib_centroid = original_poly.centroid
                rotation_angle = part_info.rotation

                # メイン桁
                if part_index < len(self.spar_positions):
                    spar_center = Point(self.spar_positions[part_index])
                    spar_diameter = self.spar_diameters[part_index]

                    # リブと同じ回転・移動を適用
                    rotated_spar_center = rotate(spar_center, rotation_angle, origin=rib_centroid)
                    final_spar_center = translate(rotated_spar_center, xoff=dx, yoff=dy)
                    final_center_coords = (final_spar_center.x, final_spar_center.y)

                    if self.circle_or_ellipse == 'circle':
                        self.drawer.draw_circle(final_center_coords, spar_diameter)
                    elif self.circle_or_ellipse == 'ellipse' and part_index < len(self.ellipse_degrees):
                        original_ellipse_degree = self.ellipse_degrees[part_index]
                        # 楕円の角度もリブの回転に合わせて変換
                        final_ellipse_degree = original_ellipse_degree + rotation_angle
                        self.drawer.draw_ellipse(final_center_coords, spar_diameter, final_ellipse_degree)

                # 後ろ側の桁 (リアスパー)
                if part_index < len(self.rear_spar_positions):
                    rear_spar_center = Point(self.rear_spar_positions[part_index])
                    rear_spar_diameter = self.rear_spar_diameters[part_index]

                    # リブと同じ回転・移動を適用
                    rotated_rear_spar_center = rotate(rear_spar_center, rotation_angle, origin=rib_centroid)
                    final_rear_spar_center = translate(rotated_rear_spar_center, xoff=dx, yoff=dy)
                    final_rear_center_coords = (final_rear_spar_center.x, final_rear_spar_center.y)

                    self.drawer.draw_circle(final_rear_center_coords, rear_spar_diameter)

            except Exception as e:
                print(f"警告: 部品 {part_info.id} の描画中にエラーが発生しました: {e}")

        # ビンの境界線を描画
        bin_height = result_packer.get_bin_height()
        bin_boundary = [
            (0, 0), (self.bin_width, 0),
            (self.bin_width, bin_height), (0, bin_height)
        ]
        self.drawer.draw_polyline(bin_boundary)

        # DXFファイルに保存
        try:
            self.drawer.save(filename)
            print(f"DXFファイルを保存しました: {filename}")
            print(f"最終高さ: {bin_height:.2f}mm")
            print(f"材料利用率: {result_packer.get_utilization():.2%}")

        except Exception as e:
            print(f"エラー: DXFファイルの保存中にエラーが発生しました: {e}")

    def generate_optimization_video(self,
                                    output_filename: str = "nesting_optimization.mp4",
                                    fps: int = 2):
        """
        GAの最適化過程を可視化する動画を単一プロセスで生成する
        """
        print(f"\nステップ5: 最適化過程の動画生成開始 ({output_filename})")

        # 1. 動画生成用に部品データと一時ディレクトリを準備
        print("  - 動画生成用の部品データを準備...")
        simplified_airfoils = self.simplify_airfoil_tuples(self.airfoil_tuple_lists, tolerance=self.simplify_tolerance)
        parts = self.create_parts_from_airfoils(simplified_airfoils)
        if not parts:
            print("エラー: 動画生成用の部品が作成できませんでした。")
            return

        frame_dir = "temp_nesting_frames_opt"
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

        # 2. 動画生成用に単一のGAインスタンスを作成
        print("  - 単一プロセスで遺伝的アルゴリズムを準備...")
        ga_params = {"population_size": 40, "generations": 50, "mutation_rate": 0.15}
        ga = GeneticAlgorithm(parts, self.bin_width, angle_step=15, **ga_params)

        # 3. NFPを計算
        ga.nfp_calculator = NFPCalculator(parts, angle_step=15, margin=self.safety_margin)
        ga.nfp_calculator.precompute_nfps()

        # --- ▼▼▼ ここからが修正の中心 ▼▼▼ ---
        # 全てのフレームで同じ図のサイズとY軸の上限を定義する
        fixed_plot_size = (10, 8)  # (横, 縦) のインチ指定
        # Y軸の上限を、材料の幅と同じくらいに余裕をもって設定
        fixed_y_axis_limit = self.bin_width

        # 動画生成専用のGA実行メソッドを呼び出す
        self.run_ga_for_video(ga, frame_dir, fixed_plot_size, fixed_y_axis_limit)
        # --- ▲▲▲ ---

        # ... (フレーム画像から動画を生成する部分は変更なし) ...
        print("  - フレーム画像を連結して動画を生成中...")
        frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])
        if not frame_files:
            # ...
            return

        with imageio.get_writer(output_filename, fps=fps, macro_block_size=1) as writer:  # macro_block_sizeを追加して警告を抑制
            for filename in frame_files:
                writer.append_data(imageio.imread(filename))

        print(f"動画を保存しました: {output_filename}")
        shutil.rmtree(frame_dir)

    def run_ga_for_video(self, ga: 'GeneticAlgorithm', frame_output_dir: str,
                         fixed_figsize: tuple, fixed_ylim: float):
        """動画生成のため、1世代ごとにフレームを保存しながらGAを実行する"""
        population = ga.initialize_population()
        best_fitness = 0
        patience = 20
        generations_without_improvement = 0

        for generation in range(ga.generations):
            print(f"  - 動画生成中: 世代 {generation + 1}/{ga.generations}")
            fitness_values = [ga.evaluate_individual(ind) for ind in population]

            # --- フレーム保存ロジック ---
            packer = ImprovedBinPacking(ga.bin_width)
            best_ind_in_gen = population[fitness_values.index(max(fitness_values))]
            for part_id, angle in best_ind_in_gen:
                part = next(p for p in ga.parts if p.id == part_id)
                rotated_part = part.rotate_to(angle)
                position = packer.find_optimal_position(rotated_part, ga.nfp_calculator)
                packer.place_part(rotated_part, position)

            frame_path = os.path.join(frame_output_dir, f"frame_{generation:04d}.png")
            title = f"Generation: {generation + 1}, Height: {packer.get_bin_height():.2f}"

            # 固定サイズのパラメータをvisualizeメソッドに渡す
            packer.visualize(save_path=frame_path, show_plot=False, title=title,
                             fixed_figsize=fixed_figsize, fixed_ylim=fixed_ylim)

            # --- GAの進化ロジック ---
            current_best_fitness = max(fitness_values)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= patience:
                break

            selected = ga.selection(population, fitness_values)
            next_population = []
            max_idx = fitness_values.index(max(fitness_values))
            next_population.append(copy.deepcopy(population[max_idx]))
            while len(next_population) < ga.population_size:
                p1, p2 = random.sample(selected, k=2)
                c1, c2 = ga.crossover(p1, p2)
                next_population.append(ga.mutate(c1))
                if len(next_population) < ga.population_size:
                    next_population.append(ga.mutate(c2))
            population = next_population


def integrate_nesting_to_wingmaker(airfoil_tuple_lists: List[List[Tuple[float, float]]],
                                  bin_width: float = 1000.0,
                                  simplify_tolerance: float = 0.5,
                                  output_filename: str = "wing_nesting_result.dxf",
                                    safety_margin: float = 0.0,
                                    generate_video: bool = False,
                                  video_filename: str = "nesting_process.mp4",
                                   spar_positions=None, rear_spar_positions=None, spar_diameters=None,
                                   rear_spar_diameters=None, ellipse_degrees=None, circle_or_ellipse='circle'
                                   ) -> dict:
    """
    Wing_Maker_Ver2からネスティング機能を呼び出すメイン関数
    
    Args:
        airfoil_tuple_lists: Wing_Makerから出力された翼型タプルリスト
        bin_width: ビンの幅（mm）
        simplify_tolerance: 近似の許容誤差
        output_filename: 出力DXFファイル名
        
    Returns:
        ネスティング結果の辞書
    """
    integrator = WingNestingIntegrator(
        airfoil_tuple_lists, bin_width, simplify_tolerance,safety_margin,
        spar_positions, rear_spar_positions, spar_diameters,
        rear_spar_diameters, ellipse_degrees, circle_or_ellipse
    )
    # ネスティング実行
    result_packer, final_solution, parts = integrator.run_nesting()
    
    if result_packer is None:
        return {"success": False, "error": "ネスティングの実行に失敗しました"}
    # 通常の可視化は表示しない
    #result_packer.visualize(show_plot=False)
    
    # DXF出力
    integrator.export_nesting_to_dxf(result_packer, output_filename)

    # ▼▼▼ 動画生成の呼び出しを追加 ▼▼▼
    if generate_video:
        # 引数を正しく指定して呼び出す
        integrator.generate_optimization_video(output_filename=video_filename)

    # 結果を辞書で返す
    return {
        "success": True,
        "bin_width": bin_width,
        "bin_height": result_packer.get_bin_height(),
        "utilization": result_packer.get_utilization(),
        "parts_count": len(result_packer.placed_parts),
        "output_file": output_filename
    }