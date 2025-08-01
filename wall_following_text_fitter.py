import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
import math


class AngleAwareWallFollowingTextFitter:
    def __init__(self, part_polygon, text_annotator):
        self.part_polygon = part_polygon
        self.text_annotator = text_annotator
        self.aspect_ratio = 5.0
        self.step_size = 2.0  # 壁に沿って移動する際のステップサイズ

    def find_wall_with_slope_analysis(self, start_point, direction, max_distance=1000):
        """
        壁を見つけて、その壁の傾きも分析する
        """
        start = Point(start_point)

        # 指定方向に長い線分を作成
        end_point = [
            start_point[0] + direction[0] * max_distance,
            start_point[1] + direction[1] * max_distance
        ]
        ray = LineString([start_point, end_point])

        # パーツの境界と交差点を探す
        intersection = ray.intersection(self.part_polygon.boundary)

        if intersection.is_empty:
            return None

        # 最も近い交点を取得
        if hasattr(intersection, 'geoms'):
            nearest_point = min(intersection.geoms,
                                key=lambda p: Point(start_point).distance(p))
            hit_point = [nearest_point.x, nearest_point.y]
        else:
            hit_point = [intersection.x, intersection.y]

        # 当たった壁のセグメントを特定し、傾きを計算
        wall_info = self._analyze_wall_segment(hit_point)

        return {
            'point': hit_point,
            'slope': wall_info['slope'],
            'wall_segment': wall_info['segment'],
            'is_vertical': wall_info['is_vertical'],
            'is_horizontal': wall_info['is_horizontal']
        }

    def _analyze_wall_segment(self, hit_point, tolerance=1e-6):
        """
        当たった点がどの壁セグメントに属するかを特定し、傾きを分析
        """
        coords = list(self.part_polygon.exterior.coords)
        hit_pt = Point(hit_point)

        min_distance = float('inf')
        closest_segment = None

        # 各セグメントとの距離を計算して最も近いものを特定
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            distance = hit_pt.distance(segment)

            if distance < min_distance:
                min_distance = distance
                closest_segment = segment

        if closest_segment is None:
            return {'slope': 0, 'segment': None, 'is_vertical': False, 'is_horizontal': True}

        # セグメントの傾きを計算
        start_coord = closest_segment.coords[0]
        end_coord = closest_segment.coords[1]

        dx = end_coord[0] - start_coord[0]
        dy = end_coord[1] - start_coord[1]

        if abs(dx) < tolerance:  # 垂直線
            slope_degrees = 90.0
            is_vertical = True
            is_horizontal = False
        elif abs(dy) < tolerance:  # 水平線
            slope_degrees = 0.0
            is_vertical = False
            is_horizontal = True
        else:
            slope_radians = math.atan2(dy, dx)
            slope_degrees = math.degrees(slope_radians)
            is_vertical = abs(abs(slope_degrees) - 90) < 15
            is_horizontal = abs(slope_degrees) < 15 or abs(abs(slope_degrees) - 180) < 15

        return {
            'slope': slope_degrees,
            'segment': closest_segment,
            'is_vertical': is_vertical,
            'is_horizontal': is_horizontal
        }

    def calculate_perpendicular_directions(self, wall_slope):
        """
        壁の傾きに対して垂直な方向を計算

        Args:
            wall_slope: 壁の傾き（度数法）

        Returns:
            tuple: (上方向ベクトル, 下方向ベクトル)
        """
        # 壁の傾きをラジアンに変換
        wall_angle_rad = math.radians(wall_slope)

        # 壁に垂直な方向を計算（壁の角度 + 90度）
        perpendicular_angle_rad = wall_angle_rad + math.pi / 2

        # 垂直方向のベクトル
        perp_dx = math.cos(perpendicular_angle_rad)
        perp_dy = math.sin(perpendicular_angle_rad)

        # 正規化
        length = math.sqrt(perp_dx * perp_dx + perp_dy * perp_dy)
        if length > 0:
            perp_dx /= length
            perp_dy /= length

        # 上下方向を決定（Yが大きい方を上とする）
        if perp_dy > 0:
            up_direction = [perp_dx, perp_dy]
            down_direction = [-perp_dx, -perp_dy]
        else:
            up_direction = [-perp_dx, -perp_dy]
            down_direction = [perp_dx, perp_dy]

        return up_direction, down_direction

    def angle_aware_algorithm(self, anchor_point):
        """
        角度を考慮した壁伝いアルゴリズム
        """
        print(f"アンカーポイント: [{anchor_point[0]:.2f}, {anchor_point[1]:.2f}]")

        # 1. 右方向に壁を探す
        wall1_info = self.find_wall_with_slope_analysis(anchor_point, [1, 0])
        if not wall1_info:
            return None

        print(
            f"壁1 (右): [{wall1_info['point'][0]:.2f}, {wall1_info['point'][1]:.2f}], 傾き: {wall1_info['slope']:.1f}°")

        # 2. 壁1の傾きに基づいて垂直方向を計算
        up_direction, down_direction = self.calculate_perpendicular_directions(wall1_info['slope'])
        print(
            f"壁1に垂直な方向 - 上: [{up_direction[0]:.3f}, {up_direction[1]:.3f}], 下: [{down_direction[0]:.3f}, {down_direction[1]:.3f}]")

        # 3. 上下両方向を試行
        best_rect = None
        best_area = 0

        for direction_name, direction_vector in [
            ('下（垂直）', down_direction),
            ('上（垂直）', up_direction)
        ]:
            print(f"\n--- {direction_name}を試行 ---")

            try:
                # 壁1から垂直方向に移動して壁を探す
                wall2_info = self.find_wall_with_slope_analysis(wall1_info['point'], direction_vector)
                if not wall2_info:
                    print(f"{direction_name}に壁が見つかりませんでした")
                    continue

                print(
                    f"壁2 ({direction_name}): [{wall2_info['point'][0]:.2f}, {wall2_info['point'][1]:.2f}], 傾き: {wall2_info['slope']:.1f}°")

                # 壁2の垂直方向を計算して左方向を決定
                wall2_up, wall2_down = self.calculate_perpendicular_directions(wall2_info['slope'])

                # 壁2から左方向を探す（複数方向を試行）
                left_candidates = [
                    [-1, 0],  # 標準的な左
                    wall2_up if wall2_up[0] < 0 else wall2_down,  # 壁2に垂直で左向き
                ]

                wall3_info = None
                for left_dir in left_candidates:
                    wall3_candidate = self.find_wall_with_slope_analysis(wall2_info['point'], left_dir)
                    if wall3_candidate:
                        wall3_info = wall3_candidate
                        break

                if not wall3_info:
                    print("左方向に壁が見つかりませんでした")
                    continue

                print(
                    f"壁3 (左): [{wall3_info['point'][0]:.2f}, {wall3_info['point'][1]:.2f}], 傾き: {wall3_info['slope']:.1f}°")

                # 4. 壁3からアンカーポイント方向に壁を探す
                # 壁3の垂直方向を計算
                wall3_up, wall3_down = self.calculate_perpendicular_directions(wall3_info['slope'])

                # アンカーポイントに近づく方向を選択
                to_anchor_x = anchor_point[0] - wall3_info['point'][0]
                to_anchor_y = anchor_point[1] - wall3_info['point'][1]

                # 壁3の垂直方向のうち、アンカーポイントに近い方を選ぶ
                dot_up = wall3_up[0] * to_anchor_x + wall3_up[1] * to_anchor_y
                dot_down = wall3_down[0] * to_anchor_x + wall3_down[1] * to_anchor_y

                if dot_up > dot_down:
                    final_direction = wall3_up
                else:
                    final_direction = wall3_down

                wall4_info = self.find_wall_with_slope_analysis(wall3_info['point'], final_direction)
                if not wall4_info:
                    print("最終方向に壁が見つかりませんでした")
                    continue

                print(
                    f"壁4: [{wall4_info['point'][0]:.2f}, {wall4_info['point'][1]:.2f}], 傾き: {wall4_info['slope']:.1f}°")

                # 5. 長方形を計算
                rect_info = self._calculate_rectangle_from_walls(anchor_point, wall1_info, wall2_info, wall3_info,
                                                                 wall4_info)
                if rect_info:
                    area = rect_info['width'] * rect_info['height']
                    print(f"長方形面積: {area:.2f}")

                    if area > best_area:
                        best_area = area
                        best_rect = rect_info
                        print(f"新しいベスト長方形: {rect_info['width']:.2f} x {rect_info['height']:.2f}")

            except Exception as e:
                print(f"{direction_name}の試行でエラー: {e}")
                continue

        return best_rect

    def _calculate_rectangle_from_walls(self, anchor_point, wall1, wall2, wall3, wall4):
        """
        4つの壁から長方形を計算（改良版）
        """
        try:
            # すべての壁の座標
            points = [anchor_point, wall1['point'], wall2['point'], wall3['point'], wall4['point']]

            # X座標とY座標の範囲を計算
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)

            # アンカーポイントを基準とした距離
            width_right = max_x - anchor_point[0]
            width_left = anchor_point[0] - min_x
            height_up = max_y - anchor_point[1]
            height_down = anchor_point[1] - min_y

            total_width = width_right + width_left
            total_height = height_up + height_down

            # 長方形が有効かチェック
            if total_width <= 0 or total_height <= 0:
                return None

            # 長方形の中心
            rect_center = [
                (min_x + max_x) / 2,
                (min_y + max_y) / 2
            ]

            # 長方形の4つの角
            corners = [
                [min_x, max_y],  # 左上
                [max_x, max_y],  # 右上
                [max_x, min_y],  # 右下
                [min_x, min_y]  # 左下
            ]

            print(
                f"計算された距離 - 右: {width_right:.2f}, 下: {height_down:.2f}, 左: {width_left:.2f}, 上: {height_up:.2f}")

            return {
                'width': total_width,
                'height': total_height,
                'center': rect_center,
                'corners': corners,
                'walls': [wall1, wall2, wall3, wall4]
            }

        except Exception as e:
            print(f"長方形計算でエラー: {e}")
            return None

    def fit_text_to_rectangle(self, rect_info, text_number=1, padding_ratio=0.85):
        """
        見つけた長方形にテキストをフィットさせる
        """
        if not rect_info:
            return None

        # アスペクト比を考慮してテキストサイズを決定
        available_width = rect_info['width'] * padding_ratio
        available_height = rect_info['height'] * padding_ratio

        text_width = available_width
        text_height = text_width / self.aspect_ratio

        if text_height > available_height:
            text_height = available_height
            text_width = text_height * self.aspect_ratio

        text_coords = [
            rect_info['center'][0] - text_width / 2,
            rect_info['center'][1] - text_height / 2
        ]

        print(f"テキストサイズ: {text_width:.2f} x {text_height:.2f}")
        print(f"テキスト位置: [{text_coords[0]:.2f}, {text_coords[1]:.2f}]")

        return self.text_annotator.draw_text_horizon(
            number=text_number,
            coordinates=text_coords,
            width=text_width,
            height=text_height
        )


# 使用例
def apply_angle_aware_wall_following_algorithm(my_part, text_annotator, drawer):
    """
    角度を考慮した壁伝いアルゴリズムを適用
    """
    fitter = AngleAwareWallFollowingTextFitter(my_part.polygon, text_annotator)

    center_point = my_part.polygon.representative_point()
    anchor_point = [center_point.x, center_point.y-3]

    print("=== 角度考慮壁伝いアルゴリズム開始 ===")

    rect_info = fitter.angle_aware_algorithm(anchor_point)

    if rect_info:
        print(f"\n=== 最適な長方形が見つかりました ===")
        print(f"サイズ: {rect_info['width']:.2f} x {rect_info['height']:.2f}")
        print(f"面積: {rect_info['width'] * rect_info['height']:.2f}")
        print(f"中心: [{rect_info['center'][0]:.2f}, {rect_info['center'][1]:.2f}]")

        text_polylines = fitter.fit_text_to_rectangle(rect_info, text_number=1)

        # 描画
        drawer.draw_polyline(list(my_part.polygon.exterior.coords))

        # デバッグ用：見つけた長方形も描画
        if rect_info['corners']:
            rect_coords = rect_info['corners'] + [rect_info['corners'][0]]
            drawer.draw_polyline(rect_coords)

        if text_polylines:
            drawer.draw_text(text_polylines)

        drawer.save('output_angle_aware_wall_following.dxf')
        print("\n処理完了。'output_angle_aware_wall_following.dxf' を確認してください。")

        return text_polylines
    else:
        print("エラー: 適切な長方形が見つかりませんでした。")
        return None

# 元のコードの最後の部分を置き換え
# final_text_polylines = apply_angle_aware_wall_following_algorithm(my_part, text_annotator, drawer)