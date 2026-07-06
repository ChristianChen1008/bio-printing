import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


# ==================== 基础工具函数 ====================

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _wrap_angle_2pi(a: float) -> float:
    a = a % (2 * math.pi)
    if a < 0:
        a += 2 * math.pi
    return a

def _angle_in_sweep(theta, a0, a1, ccw=True, eps=1e-9):
    if ccw:
        sweep  = _wrap_angle_2pi(a1 - a0)
        offset = _wrap_angle_2pi(theta - a0)
        return offset <= sweep + eps
    else:
        sweep  = _wrap_angle_2pi(a0 - a1)
        offset = _wrap_angle_2pi(a0 - theta)
        return offset <= sweep + eps


# ==================== 抽象基类 ====================

class GeometricSegment(ABC):
    @abstractmethod
    def distance(self, point_xy: np.ndarray) -> float: pass

    @abstractmethod
    def sample(self, step_mm: float = 1.0) -> np.ndarray: pass

    @abstractmethod
    def get_endpoints(self) -> Tuple[np.ndarray, np.ndarray]: pass


# ==================== 直线段 ====================

class LineSegment(GeometricSegment):
    """直线段，由起点和终点定义"""

    def __init__(self, start_xy, end_xy):
        self.start     = np.array(start_xy, dtype=float)
        self.end       = np.array(end_xy,   dtype=float)
        self.direction = self.end - self.start
        self.length    = _norm(self.direction)
        self.unit_dir  = (self.direction / self.length
                          if self.length > 1e-12 else np.zeros(2))

    def distance(self, point_xy):
        p = np.array(point_xy, dtype=float)
        if self.length < 1e-12:
            return _norm(p - self.start)
        t = np.clip(np.dot(p - self.start, self.unit_dir), 0.0, self.length)
        return _norm(p - (self.start + t * self.unit_dir))

    def sample(self, step_mm=1.0):
        if self.length < 1e-9:
            return np.array([self.start])
        n = max(2, int(self.length / step_mm) + 1)
        return np.array([self.start + t * self.direction
                         for t in np.linspace(0, 1, n)], dtype=float)

    def get_endpoints(self):
        return self.start.copy(), self.end.copy()


# ==================== 圆弧段 ====================

class ArcSegment(GeometricSegment):
    """
    圆弧段
    :param center_xy:       圆心坐标
    :param radius_mm:       半径（毫米）
    :param start_angle_rad: 起始角度（弧度）
    :param end_angle_rad:   终止角度（弧度）
    :param ccw:             True=逆时针，False=顺时针
    """

    def __init__(self, center_xy, radius_mm, start_angle_rad,
                 end_angle_rad, ccw=True):
        self.center      = np.array(center_xy, dtype=float)
        self.radius      = float(radius_mm)
        self.start_angle = float(start_angle_rad)
        self.end_angle   = float(end_angle_rad)
        self.ccw         = bool(ccw)
        if ccw:
            self.sweep_angle = _wrap_angle_2pi(end_angle_rad - start_angle_rad)
        else:
            self.sweep_angle = _wrap_angle_2pi(start_angle_rad - end_angle_rad)
        if self.sweep_angle < 1e-9:
            self.sweep_angle = 2 * math.pi
        self.arc_length     = abs(self.radius) * self.sweep_angle
        self.is_full_circle = abs(self.sweep_angle - 2 * math.pi) < 1e-6

    def _pt(self, a):
        return self.center + self.radius * np.array([math.cos(a), math.sin(a)])

    def distance(self, point_xy):
        p = np.array(point_xy, dtype=float)
        v = p - self.center
        d = _norm(v)
        if d < 1e-12:
            return abs(self.radius)
        theta = math.atan2(v[1], v[0])
        if self.is_full_circle or _angle_in_sweep(
                theta, self.start_angle, self.end_angle, self.ccw):
            return abs(d - self.radius)
        p0, p1 = self.get_endpoints()
        return min(_norm(p - p0), _norm(p - p1))

    def sample(self, step_mm=1.0):
        if self.arc_length < 1e-9:
            return np.array([self._pt(self.start_angle)])
        n = max(2, int(self.arc_length / step_mm) + 1)
        pts = []
        for i in range(n):
            t = i / (n - 1)
            a = (self.start_angle + t * self.sweep_angle if self.ccw
                 else self.start_angle - t * self.sweep_angle)
            pts.append(self._pt(a))
        return np.array(pts, dtype=float)

    def get_endpoints(self):
        return self._pt(self.start_angle), self._pt(self.end_angle)


# ==================== 轨迹类 ====================

class Trajectory:
    """
    轨迹类，由多个几何段（直线/圆弧）组成

    使用示例：
        traj = Trajectory("圆形")
        traj.add_arc([0, 0], 90, 0, 2*math.pi)     # 整圆
        traj.add_line([0, 90], [0, -90])             # 直线
    """

    def __init__(self, name="未命名"):
        self.name     = name
        self.segments: List[GeometricSegment] = []
        self.z_height = None
        # 新增：3D打印相关属性
        self.is_3d = False   # 是否用于多层3D打印
        self.layers = 1      # 打印层数（仅当 is_3d=True 时有效）
        # 预生成的路径点（3D模式包含层间过渡），None 表示未预生成
        self.generated_points: np.ndarray | None = None
        # 层分组信息：每层记录 (print_start, print_end, trans_start, trans_end, z)
        # 索引指向 self.generated_points，trans可为(-1,-1)表示无过渡段
        self.layer_segments: list = []

    def get_layer_groups(self):
        """
        将 generated_points 按层拆分为打印段和过渡段。

        返回: list of dict
            [
                {
                    'layer': 0,              # 层号（从0开始）
                    'z': 0.0,               # 该层工作高度
                    'print_pts': [...],      # 挤出段路径点 [[x,y,z], ...]
                    'transition_pts': [...], # 过渡段路径点（无挤出）[[x,y,z], ...]
                },
                ...
            ]
            最后一层的 transition_pts 为空列表。
            非3D模式或无预生成点时返回空列表。
        """
        if not self.generated_points or not self.layer_segments:
            # 兼容旧版：无 layer_segments 时回退为整体路径
            if self.generated_points is not None and len(self.generated_points) > 0:
                return [{
                    'layer': 0,
                    'z': self.generated_points[0][2] if len(self.generated_points[0]) > 2 else 0,
                    'print_pts': self.generated_points,
                    'transition_pts': [],
                }]
            return []

        groups = []
        pts = self.generated_points
        for seg in self.layer_segments:
            p_start, p_end, t_start, t_end, z = seg
            print_pts = pts[p_start:p_end]
            if t_start >= 0 and t_end >= 0 and t_end > t_start:
                trans_pts = pts[t_start:t_end]
            else:
                trans_pts = []
            groups.append({
                'layer': len(groups),
                'z': z,
                'print_pts': print_pts,
                'transition_pts': trans_pts,
            })
        return groups

    def add_line(self, start, end):
        """
        添加直线段
        :param start: 起点 [x, y]
        :param end:   终点 [x, y]
        """
        self.segments.append(LineSegment(start, end))

    def add_arc(self, center, radius, start_angle, end_angle, ccw=True):
        """
        添加圆弧段
        :param center:      圆心 [x, y]
        :param radius:      半径（毫米）
        :param start_angle: 起始角度（弧度）
        :param end_angle:   终止角度（弧度）
        :param ccw:         True=逆时针
        """
        self.segments.append(ArcSegment(center, radius,
                                        start_angle, end_angle, ccw))

    def generate_path_points(self, step_mm=5.0):
        """
        生成路径采样点
        :param step_mm: 采样步长（毫米）
        :return: numpy数组，shape=(N, 2)
        """
        all_pts = []
        for seg in self.segments:
            pts = seg.sample(step_mm)
            if all_pts and _norm(pts[0] - all_pts[-1]) < 1e-6:
                pts = pts[1:]
            all_pts.extend(pts.tolist())
        return np.array(all_pts, dtype=float) if all_pts else np.array([])

    def get_total_length(self):
        """返回轨迹总长度（毫米）"""
        total = 0.0
        for seg in self.segments:
            if isinstance(seg, LineSegment):
                total += seg.length
            elif isinstance(seg, ArcSegment):
                total += seg.arc_length
        return total

    def calculate_errors(self, actual_xy):
        """
        计算实际轨迹相对于理想轨迹的误差
        :param actual_xy: 实际轨迹点，shape=(N, 2)
        :return: dict，包含 rmse/mean/max/std/distances
        """
        if len(actual_xy) == 0 or not self.segments:
            return dict(rmse=float("nan"), mean=float("nan"),
                        max=float("nan"),  std=float("nan"),
                        distances=np.array([]))
        dists = np.array([min(s.distance(p) for s in self.segments)
                          for p in actual_xy])
        return dict(rmse=float(np.sqrt(np.mean(dists**2))),
                    mean=float(np.mean(dists)),
                    max =float(np.max(dists)),
                    std =float(np.std(dists)),
                    distances=dists)

    def __len__(self):
        return len(self.segments)

    def __str__(self):
        return (f"轨迹 '{self.name}': {len(self.segments)} 段, "
                f"总长度 {self.get_total_length():.1f} mm" +
                (f", 3D打印层数={self.layers}" if self.is_3d else ""))


# ==================== 预置轨迹工厂 ====================

class TrajectoryFactory:
    """
    常用轨迹生成工厂类

    使用示例：
        traj = TrajectoryFactory.circle([0,0], 90)
        traj = TrajectoryFactory.chord_diagram([0,0], 90, num_chords=25)
        traj = TrajectoryFactory.rectangle([0,0], 100, 80, is_3d=True, layers=5)
    """

    @staticmethod
    def circle(center, radius, name="圆形", is_3d=False, layers=1):
        """
        整圆轨迹
        :param center: 圆心 [x, y]
        :param radius: 半径（毫米）
        :param is_3d:   是否用于3D打印（多层）
        :param layers:  打印层数（仅当 is_3d=True 时有效）
        """
        traj = Trajectory(name)
        traj.add_arc(center, radius, 0, 2 * math.pi, ccw=True)
        traj.is_3d = is_3d
        traj.layers = layers
        return traj

    @staticmethod
    def rectangle(
        center,
        width,
        height,
        line_width=1.0,
        start_corner="bottom_left",
        start_axis="y",
        name="矩形",
        is_3d=False,
        layers=1,
        layer_height=1.0,
        corner_points=5,
    ):
        """
        填充矩形面的蛇形轨迹，支持3D多层打印。

        参数说明
        --------
        center        : 矩形中心点 [x, y] 或 [x, y, z]（z为第一层高度，默认0）
        width         : 矩形宽度（毫米），对应 X 方向
        height        : 矩形高度（毫米），对应 Y 方向
        line_width    : 填充线宽（毫米），即相邻蛇形线间距，默认 1.0
        start_corner  : 起始角，可选 "bottom_left"/"bottom_right"/"top_left"/"top_right"
        start_axis    : 蛇形扫描方向，"y"=沿Y轴来回（竖向线条）"x"=沿X轴来回（横向线条）
        name          : 轨迹名称
        is_3d         : 是否开启多层3D打印
        layers        : 打印层数（is_3d=True 时有效）
        layer_height  : 每层抬升高度（毫米），默认 1.0
        corner_points : 每个拐角处额外插入的点数（不含端点），默认 5

        返回
        --------
        list of list  : 所有轨迹点，每个点为 [x, y] 或 [x, y, z]
                        3D模式下每层末尾自动插入回原点→抬升→新层起点的过渡路径。
        """

        # ---------- 基础边界计算 ----------
        cx = float(center[0])
        cy = float(center[1])
        cz = float(center[2]) if len(center) > 2 else 0.0

        hw = width  / 2.0
        hh = height / 2.0

        x_min, x_max = cx - hw, cx + hw
        y_min, y_max = cy - hh, cy + hh

        use_3d = is_3d or len(center) > 2

        def _pt(x, y, z):
            return [x, y, z] if use_3d else [x, y]

        def _corner_pts(p_from, p_to, z, n=corner_points):
            """在拐角两端点之间插入 n 个线性过渡点（包含两端点）。"""
            pts = []
            for i in range(n + 2):          # 包含起点和终点
                t = i / (n + 1)
                pts.append(_pt(
                    p_from[0] + t * (p_to[0] - p_from[0]),
                    p_from[1] + t * (p_to[1] - p_from[1]),
                    z
                ))
            return pts

        def _build_one_layer(z_current, layer_idx):
            """生成单层蛇形填充点列表。"""
            pts = []

            if start_axis == "y":
                # 沿 X 方向排列扫描线，每条线沿 Y 方向运动
                xs = []
                x = x_min
                while x <= x_max + 1e-9:
                    xs.append(min(x, x_max))
                    x += line_width
                if abs(xs[-1] - x_max) > 1e-9:
                    xs.append(x_max)

                # 根据起始角决定 X 方向和 Y 起始端
                if start_corner in ("bottom_right", "top_right"):
                    xs = xs[::-1]
                go_up = start_corner in ("bottom_left", "bottom_right")

                for i, xi in enumerate(xs):
                    if i % 2 == 0:
                        y_start = y_min if go_up else y_max
                        y_end   = y_max if go_up else y_min
                    else:
                        y_start = y_max if go_up else y_min
                        y_end   = y_min if go_up else y_max

                    # 直线段两端点（含端点）
                    n_line = max(2, int(abs(y_end - y_start) / line_width) + 1)
                    for j in range(n_line):
                        t = j / (n_line - 1)
                        pts.append(_pt(xi, y_start + t * (y_end - y_start), z_current))

                    # 拐角过渡（最后一列不需要）
                    if i < len(xs) - 1:
                        next_xi = xs[i + 1]
                        corner_from = [xi,      y_end]
                        corner_to   = [next_xi, y_end]
                        # 跳过已加入的 corner_from（就是上面直线的终点）
                        extra = _corner_pts(corner_from, corner_to, z_current)
                        pts.extend(extra[1:])   # 去掉重复的起点

            else:  # start_axis == "x"
                # 沿 Y 方向排列扫描线，每条线沿 X 方向运动
                ys = []
                y = y_min
                while y <= y_max + 1e-9:
                    ys.append(min(y, y_max))
                    y += line_width
                if abs(ys[-1] - y_max) > 1e-9:
                    ys.append(y_max)

                if start_corner in ("top_left", "top_right"):
                    ys = ys[::-1]
                go_right = start_corner in ("bottom_left", "top_left")

                for i, yi in enumerate(ys):
                    if i % 2 == 0:
                        x_start = x_min if go_right else x_max
                        x_end   = x_max if go_right else x_min
                    else:
                        x_start = x_max if go_right else x_min
                        x_end   = x_min if go_right else x_max

                    n_line = max(2, int(abs(x_end - x_start) / line_width) + 1)
                    for j in range(n_line):
                        t = j / (n_line - 1)
                        pts.append(_pt(x_start + t * (x_end - x_start), yi, z_current))

                    if i < len(ys) - 1:
                        next_yi = ys[i + 1]
                        corner_from = [x_end, yi]
                        corner_to   = [x_end, next_yi]
                        extra = _corner_pts(corner_from, corner_to, z_current)
                        pts.extend(extra[1:])

            return pts

        # ---------- 主逻辑 ----------
        traj = Trajectory(name)
        traj.is_3d  = is_3d
        traj.layers = layers

        all_points = []
        total_layers = layers if is_3d else 1

        for layer_idx in range(total_layers):
            z_current = cz + layer_idx * layer_height

            # 记录该层打印段的起始索引
            print_start = len(all_points)

            layer_pts = _build_one_layer(z_current, layer_idx)
            all_points.extend(layer_pts)

            # 记录该层打印段的结束索引
            print_end = len(all_points)

            trans_start = -1
            trans_end   = -1

            # 3D 模式：每层结束后插入"抬升Z轴→回原点"的过渡路径
            # 顺序：先垂直抬升（避免喷嘴刮伤已打印层），再水平回退到原点
            if is_3d and layer_idx < total_layers - 1:
                last_pt    = layer_pts[-1]
                z_next     = cz + (layer_idx + 1) * layer_height
                origin_xy  = [cx - hw, cy - hh]  # 矩形原点（左下角）

                # 记录过渡段起始索引
                trans_start = len(all_points)

                # ① 先在当前位置垂直抬升Z轴（避免水平移动时刮伤已打印层）
                lift_n = max(2, int(layer_height / (line_width * 0.5)) + 1)
                for k in range(1, lift_n + 1):
                    t = k / lift_n
                    all_points.append(_pt(
                        last_pt[0],
                        last_pt[1],
                        z_current + t * layer_height
                    ))

                # ② 抬升后在高层水平移动到原点（安全高度，不会碰到已打印层）
                retreat = _corner_pts(
                    [last_pt[0], last_pt[1]],
                    origin_xy,
                    z_next
                )
                all_points.extend(retreat[1:])           # 去掉重复起点

                # 记录过渡段结束索引
                trans_end = len(all_points)

            # 保存该层的分段信息
            traj.layer_segments.append(
                (print_start, print_end, trans_start, trans_end, z_current)
            )

            # 同时也把路径加入 Trajectory 对象（可选，保留原有接口）
            if len(layer_pts) >= 2:
                for i in range(len(layer_pts) - 1):
                    p0 = layer_pts[i][:2]
                    p1 = layer_pts[i+1][:2]
                    traj.add_line(p0, p1)

        traj.generated_points = all_points
        return traj

    @staticmethod
    def rectangle_outline(
        center,
        length_x,
        width_y,
        step_mm=1.0,
        name="矩形轮廓",
        is_3d=False,
        layers=1,
        layer_height=1.0,
    ):
        """
        矩形轮廓轨迹（仅外围边框，不填充内部）

        参数说明
        --------
        center        : 矩形中心点 [x, y] 或 [x, y, z]（z为第一层高度，默认0）
        length_x      : X轴方向长度（mm）
        width_y       : Y轴方向宽度（mm）
        step_mm       : 采样步长（mm），默认 1.0
        name          : 轨迹名称
        is_3d         : 是否开启多层3D打印
        layers        : 打印层数（is_3d=True 时有效）
        layer_height  : 每层抬升高度（mm），默认 1.0

        返回
        --------
        Trajectory : 包含矩形轮廓的轨迹对象
        """
        cx = float(center[0])
        cy = float(center[1])
        cz = float(center[2]) if len(center) > 2 else 0.0

        hx = length_x / 2.0
        hy = width_y  / 2.0

        # 四个角点（左下 → 右下 → 右上 → 左上）
        corners = [
            [cx - hx, cy - hy],   # 左下
            [cx + hx, cy - hy],   # 右下
            [cx + hx, cy + hy],   # 右上
            [cx - hx, cy + hy],   # 左上
        ]

        traj = Trajectory(name)
        traj.is_3d  = is_3d
        traj.layers = layers

        use_3d = is_3d or len(center) > 2

        def _pt(x, y, z):
            return [x, y, z] if use_3d else [x, y]

        all_points = []
        total_layers = layers if is_3d else 1

        for layer_idx in range(total_layers):
            z_current = cz + layer_idx * layer_height

            # 记录该层打印段的起始索引
            print_start = len(all_points)

            # 四条边，每条边等距采样
            for i in range(4):
                p_start = corners[i]
                p_end   = corners[(i + 1) % 4]

                dx = p_end[0] - p_start[0]
                dy = p_end[1] - p_start[1]
                seg_len = math.sqrt(dx * dx + dy * dy)
                n = max(2, int(seg_len / step_mm) + 1)

                for j in range(n):
                    # 最后一条边的最后一个点与起点重合，跳过避免重复
                    if i == 3 and j == n - 1:
                        continue
                    t = j / (n - 1)
                    all_points.append(_pt(
                        p_start[0] + t * dx,
                        p_start[1] + t * dy,
                        z_current
                    ))

            # 记录该层打印段的结束索引
            print_end = len(all_points)
            trans_start = -1
            trans_end   = -1

            # 将四条边加入 Trajectory 几何段
            for i in range(4):
                traj.add_line(corners[i], corners[(i + 1) % 4])

            # 3D 模式：层间抬升过渡
            if is_3d and layer_idx < total_layers - 1:
                z_next = cz + (layer_idx + 1) * layer_height
                first_pt_next = [cx - hx, cy - hy, z_next]

                # 记录过渡段起始索引
                trans_start = len(all_points)

                lift_n = max(2, int(layer_height / (step_mm * 0.5)) + 1)
                for k in range(1, lift_n + 1):
                    t = k / lift_n
                    all_points.append(_pt(
                        first_pt_next[0],
                        first_pt_next[1],
                        z_current + t * layer_height
                    ))

                # 记录过渡段结束索引
                trans_end = len(all_points)

            # 保存该层的分段信息
            traj.layer_segments.append(
                (print_start, print_end, trans_start, trans_end, z_current)
            )

        traj.generated_points = all_points
        return traj

    @staticmethod
    def line(start, end, step_mm=1.0, name="直线"):
        """
        直线轨迹

        参数说明
        --------
        start   : 起点 [x, y]
        end     : 终点 [x, y]
        step_mm : 采样步长（毫米），默认 1.0
        name    : 轨迹名称

        返回
        --------
        Trajectory : 包含一条直线段的轨迹对象
        """
        traj = Trajectory(name)
        traj.add_line(start, end)

        pts = traj.generate_path_points(step_mm)
        traj.generated_points = pts.tolist()
        return traj

    @staticmethod
    def chord_diagram(center, radius, num_chords=25,
                      name="弦图", is_3d=False, layers=1):
        """
        弦图轨迹（蛇形弦线填充圆形）
        :param center:     圆心 [x, y] 或 [x, y, z]
        :param radius:     圆半径（毫米）
        :param num_chords: 弦线数量
        :param is_3d:       是否用于3D打印（多层）
        :param layers:      打印层数（仅当 is_3d=True 时有效）
        """
        traj = Trajectory(name)
        cx, cy = center[0], center[1]
        xs = np.linspace(-radius * 0.98, radius * 0.98, num_chords)

        tops, bottoms = [], []
        for dx in xs:
            dy = math.sqrt(max(0, radius**2 - dx**2))
            tops.append([cx + dx, cy + dy])
            bottoms.append([cx + dx, cy - dy])

        for i in range(num_chords):
            if i % 2 == 0:
                traj.add_line(bottoms[i], tops[i])
                if i < num_chords - 1:
                    traj.add_line(tops[i], tops[i + 1])
            else:
                traj.add_line(tops[i], bottoms[i])
                if i < num_chords - 1:
                    traj.add_line(bottoms[i], bottoms[i + 1])

        # 保留原有的 z_height 属性（如果 center 有 Z 坐标）
        if len(center) > 2:
            traj.z_height = center[2]

        traj.is_3d = is_3d
        traj.layers = layers
        return traj