import time
import threading
import numpy as np

try:
    import rtde_control as rc
    import rtde_receive as rr
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("警告: rtde_control/rtde_receive 未安装，机械臂功能不可用")


class UR3Controller:
    """
    UR3机械臂控制类
    通过RTDE接口控制UR3机械臂运动

    坐标系说明：
        所有坐标单位均为毫米(mm)
        origin_offset 为本地坐标系相对机器人坐标系的偏移
    """

    DEFAULT_IP    = "169.254.45.1"
    DEFAULT_SPEED = 5     # mm/s
    DEFAULT_ACCEL = 1000    # mm/s²
    SAMPLE_RATE   = 10      # Hz，位置采样频率

    def __init__(self, robot_ip=DEFAULT_IP):
        if not RTDE_AVAILABLE:
            raise ImportError("请先安装 rtde_control 和 rtde_receive")
        self.robot_ip     = robot_ip
        self.speed        = self.DEFAULT_SPEED
        self.acceleration = self.DEFAULT_ACCEL
        # 本地坐标系偏移（毫米+弧度）
        self.origin_offset = np.array([-126, -382, 52, 0.009, -3.123, -0.078])
        self.sampled_actual = []
        self._recording     = False
        self._record_thread = None
        self._connect()
        time.sleep(1)
        print(f"✓ 机械臂连接成功。TCP当前位置: {self.get_tcp()[:3]}")
        print(f"✓ 速度: {self.speed} mm/s，加速度: {self.acceleration} mm/s²")

    # ==================== 连接管理 ====================

    def _connect(self):
        try:
            self.rtde_c = rc.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rr.RTDEReceiveInterface(self.robot_ip)
        except Exception as e:
            raise ConnectionError(f"无法连接机械臂 {self.robot_ip}: {e}")

    def disconnect(self):
        """断开机械臂连接"""
        if self._recording:
            self.stop_recording()
        try:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception:
            pass
        print("✓ 机械臂已断开连接")

    # ==================== 坐标转换 ====================

    def _to_robot(self, p):
        """本地坐标 → 机器人坐标"""
        return np.array(p, dtype=float) + self.origin_offset

    def _to_local(self, p):
        """机器人坐标 → 本地坐标"""
        return np.array(p, dtype=float) - self.origin_offset

    # ==================== 位置读取 ====================

    def get_tcp(self):
        """
        读取当前TCP位置（本地坐标系，单位mm）
        返回: [x, y, z, rx, ry, rz]
        """
        m  = self.rtde_r.getActualTCPPose()
        mm = np.array([m[0]*1000, m[1]*1000, m[2]*1000, m[3], m[4], m[5]])
        return self._to_local(mm)

    # ==================== 运动控制 ====================

    def move_to_point(self, target) -> bool:
        """
        移动到目标点（阻塞直到到达）
        :param target: [x,y,z] 或 [x,y,z,rx,ry,rz]，单位mm
        """
        target = np.array(target, dtype=float)
        if target.shape[0] == 3:
            target = np.concatenate([target, self.get_tcp()[3:]])
        rm = self._to_robot(target).copy()
        rm[:3] /= 1000.0
        try:
            self.rtde_c.moveL(rm.tolist(),
                              self.speed / 1000.0,
                              self.acceleration / 1000.0)
            return True
        except Exception as e:
            print(f"运动指令失败: {e}")
            return False

    def move_path(self, path_3d, blend_radius=0.002) -> bool:
        """
        按路径点列表运动（混合运动，更平滑）
        :param path_3d: 路径点列表，每个点为 [x,y,z]，单位mm
        :param blend_radius: 混合半径，单位m
        """
        cur = self.get_tcp()
        wps = []
        for pt in path_3d:
            rp = self._to_robot(
                np.concatenate([pt, cur[3:]]) if len(pt) == 3
                else np.array(pt, dtype=float))
            rm = rp.copy()
            rm[:3] /= 1000.0
            wps.append(rm.tolist() +
                       [self.speed / 1000.0,
                        self.acceleration / 1000.0,
                        blend_radius])
        try:
            self.rtde_c.moveL(wps)
            return True
        except Exception as e:
            print(f"路径执行失败: {e}")
            return False

    # ==================== 位置记录 ====================

    def start_recording(self):
        """开始记录TCP实际位置"""
        if self._recording:
            return
        self._recording = True
        self.sampled_actual = []
        dt = 1.0 / self.SAMPLE_RATE

        def _worker():
            while self._recording:
                self.sampled_actual.append(self.get_tcp()[:3].copy())
                time.sleep(dt)

        self._record_thread = threading.Thread(target=_worker, daemon=True)
        self._record_thread.start()
        print("▶ 开始记录位置数据...")

    def stop_recording(self):
        """停止记录，返回采集到的位置数组"""
        self._recording = False
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        print(f"◼ 记录完成，共采集 {len(self.sampled_actual)} 个点")
        return np.array(self.sampled_actual, dtype=float)

    # ==================== 轨迹执行 ====================

    def execute_trajectory(self, trajectory, z_height=None,
                           use_path_mode=True, blend_radius=0.002):
        """
        执行轨迹对象
        :param trajectory: Trajectory 实例
        :param z_height:   工作高度(mm)，None则使用当前高度
        :param use_path_mode: True=路径混合模式，False=逐点模式
        :param blend_radius:  混合半径(m)
        :return: 实际采集到的位置数组
        """
        if not trajectory.segments:
            print("错误：轨迹为空")
            return None

        print(f"\n{'='*60}\n正在执行: {trajectory}\n{'='*60}")
        print(f"运动参数: 速度 {self.speed} mm/s，加速度 {self.acceleration} mm/s²")

        if z_height is None:
            z_height = float(self.get_tcp()[2])
        print(f"工作高度 Z: {z_height:.2f} mm")

        path_2d = trajectory.generate_path_points(step_mm=5.0)
        path_3d = np.column_stack([path_2d, np.full(len(path_2d), z_height)])
        print(f"路径点数: {len(path_3d)}，总长度: {trajectory.get_total_length():.1f} mm")
        print(f"预计时间: {trajectory.get_total_length() / self.speed:.2f} 秒")

        print(f"移动到起点: {path_3d[0]}")
        self.move_to_point(path_3d[0])
        time.sleep(0.8)

        self.start_recording()
        if use_path_mode:
            self.move_path(path_3d[1:], blend_radius=blend_radius)
        else:
            for i, pt in enumerate(path_3d[1:], 1):
                if i % 50 == 0:
                    print(f"  进度: {i}/{len(path_3d)-1}")
                self.move_to_point(pt)
                time.sleep(0.05)

        actual = self.stop_recording()
        print("✓ 轨迹执行完成")
        return actual
    
if __name__ == "__main__":
    controller = UR3Controller("169.254.45.1")
    controller.move_to_point([100,100,1.5])
    controller.disconnect()
    