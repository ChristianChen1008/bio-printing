# -*- coding: utf-8 -*-
"""
3D打印综合控制台 — 窗口逻辑
整合机械臂、喷头电机、轨迹生成，提供完整的打印控制界面。

当前窗口已支持两种打印模式：
1. 单条轨迹：沿用原有矩形、圆形、弦图、直线等单路径打印
2. 复合程序：按步骤执行多个轨迹和过渡动作

当前复合程序能力包括：
- 预览复合路径总路径
- 执行 travel / print_trajectory / lift / retract 四类步骤
- 在界面中通过“单条轨迹 / 复合程序”切换模式
- 通过预设组合路径执行第一版复杂工艺流程
"""

import sys
import os
import time
import threading
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

from MINE_ui import Ui_PrintWindow

# 导入底层控制模块 — 自动定位到 python/ 根目录
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from 控制代码.UR3Controller import UR3Controller
from 控制代码.Motor import Motor
from 控制代码.PathProgram import (
    PathProgram,
    STEP_TYPE_LIFT,
    STEP_TYPE_PRINT_TRAJECTORY,
    STEP_TYPE_RETRACT,
    STEP_TYPE_TRAVEL,
    build_rectangle_circle_program,
)
from 控制代码.Trajectory import TrajectoryFactory


# ==================== 工具函数 ====================

def _safe_float(text, default=0.0):
    try:
        return float(text)
    except (ValueError, TypeError):
        return default


def _safe_int(text, default=0):
    try:
        return int(text)
    except (ValueError, TypeError):
        return default


def _trajectory_to_path_3d(traj, default_z):
    """将单条 Trajectory 统一转换为 (N, 3) 预览点。"""
    if traj.generated_points is not None:
        raw = np.array(traj.generated_points, dtype=float)
        if raw.ndim == 2 and raw.shape[1] >= 3:
            return raw[:, :3]
        if raw.ndim == 2 and raw.shape[1] == 2:
            return np.column_stack([raw, np.full(len(raw), default_z)])
        return np.empty((0, 3), dtype=float)

    path_2d = traj.generate_path_points(step_mm=5.0)
    if len(path_2d) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack([path_2d, np.full(len(path_2d), default_z)])


def _map_corner_value(corner_value):
    return {
        "bl": "bottom_left",
        "br": "bottom_right",
        "tl": "top_left",
        "tr": "top_right",
    }.get(corner_value, corner_value)


# ==================== 打印工作线程 ====================

class PrintWorker(QThread):
    """后台执行打印流程，避免阻塞 UI"""
    log_signal = pyqtSignal(str)          # 日志消息
    state_signal = pyqtSignal(str)        # 状态变化
    finished_signal = pyqtSignal(bool)    # 完成信号 (True=成功, False=失败或中止)

    def __init__(self, params, arm, motor):
        super().__init__()
        self.params = params
        self._abort = False
        self.arm = arm    # 暴露给主线程做紧急停止
        self.motor = motor
        self._current_point = None

    def _abort_if_requested(self, stop_motor=False):
        if not self._abort:
            return
        if stop_motor and self.motor is not None:
            try:
                self.motor.stop(emergency=False)
            except Exception:
                pass
        raise InterruptedError("打印已中止")

    def _sleep_with_abort(self, seconds, stop_motor=False):
        deadline = time.time() + max(float(seconds), 0.0)
        while time.time() < deadline:
            self._abort_if_requested(stop_motor=stop_motor)
            time.sleep(min(0.05, max(deadline - time.time(), 0.0)))

    def _move_to_point(self, point_xyz, *, log_message=None):
        point = np.array(point_xyz, dtype=float)[:3]
        if log_message:
            self.log_signal.emit(log_message)
        self._run_arm_motion(self.arm.move_to_point, point)
        self._current_point = point
        self._abort_if_requested()

    def _move_path(self, path_points, *, blend_radius, log_message=None, stop_motor_on_abort=False):
        path_3d = np.array(path_points, dtype=float)
        if len(path_3d) == 0:
            return
        if log_message:
            self.log_signal.emit(log_message)
        self._run_arm_motion(self.arm.move_path, path_3d, blend_radius=blend_radius)
        self._current_point = np.array(path_3d[-1], dtype=float)[:3]
        self._abort_if_requested(stop_motor=stop_motor_on_abort)

    def _build_single_trajectory(self):
        p = self.params
        traj_type = p["path_type"]
        cx, cy = p["center_x"], p["center_y"]
        is_3d = p["is_3d"]
        layers = p["layers"] if is_3d else 1
        corner = _map_corner_value(p.get("corner", "bottom_left"))

        if traj_type == "矩形填充":
            return TrajectoryFactory.rectangle(
                center=[cx, cy, p["work_z"]],
                width=p["width"],
                height=p["height"],
                line_width=p["line_width"],
                start_corner=corner,
                name="矩形",
                is_3d=is_3d,
                layers=layers,
                layer_height=p["layer_height"],
            )
        if traj_type == "矩形轮廓":
            return TrajectoryFactory.rectangle_outline(
                center=[cx, cy, p["work_z"]],
                length_x=p["width"],
                width_y=p["height"],
                step_mm=1.0,
                name="矩形轮廓",
                is_3d=is_3d,
                layers=layers,
                layer_height=p["layer_height"],
            )
        if traj_type == "圆形":
            return TrajectoryFactory.circle(
                center=[cx, cy],
                radius=p["radius"],
                name="圆形",
                is_3d=is_3d,
                layers=layers,
            )
        if traj_type == "弦图":
            return TrajectoryFactory.chord_diagram(
                center=[cx, cy],
                radius=p["radius"],
                num_chords=p["num_chords"],
                name="弦图",
                is_3d=is_3d,
                layers=layers,
            )
        if traj_type == "直线":
            return TrajectoryFactory.line(
                start=[p["line_x1"], p["line_y1"]],
                end=[p["line_x2"], p["line_y2"]],
                step_mm=p["line_step"],
                name="直线",
            )
        raise ValueError(f"未知轨迹类型: {traj_type}")

    def _trajectory_to_execution_path(self, traj, default_z=None):
        return _trajectory_to_path_3d(traj, self.params["work_z"] if default_z is None else default_z)

    def _resolve_program_step_z(self, step):
        if "work_z" in step.params:
            return float(step.params["work_z"])

        traj = step.params.get("trajectory")
        traj_z = getattr(traj, "z_height", None)
        if traj_z is not None:
            return float(traj_z)

        return float(self.params["work_z"])

    def _prime_and_start_extrusion(self):
        p = self.params
        motor = self.motor

        prime_pulses = p["prime"]
        if prime_pulses > 0:
            self.log_signal.emit(f"预挤出 {prime_pulses} 脉冲...")
            motor.set_speed(p["ext_rpm"])
            motor.forward(prime_pulses)
            self._abort_if_requested(stop_motor=True)

        ext_rpm = p["ext_rpm"]
        delay = p["ext_delay"]
        if ext_rpm > 0 or delay > 0:
            self.log_signal.emit(f"持续挤出 {ext_rpm} r/min, 等待 {delay} 秒")
            motor.set_speed(ext_rpm)
            motor.move(CW=False)
            if delay > 0:
                self._sleep_with_abort(delay, stop_motor=True)
        else:
            self.log_signal.emit("跳过持续挤出（速度=0 且 延迟=0）")
        self._abort_if_requested(stop_motor=True)

    def _stop_and_retract(self, pulses=None, speed=50, *, pause_seconds=0.3):
        retract_pulses = self.params["retract"] if pulses is None else int(pulses)
        self.log_signal.emit("停止喷头...")
        self.motor.stop(emergency=False)
        self.log_signal.emit(f"回抽 {retract_pulses} 脉冲")
        self.motor.backward(retract_pulses, speed=speed)
        if pause_seconds > 0:
            self._sleep_with_abort(pause_seconds)

    def _execute_print_path(self, path_3d, *, retract_after):
        p = self.params
        path_3d = np.array(path_3d, dtype=float)
        if len(path_3d) < 2:
            raise RuntimeError("路径点不足，无法执行")

        self._move_to_point(path_3d[0], log_message=f"移动到起点: {path_3d[0]}")
        self._sleep_with_abort(0.5)
        self._prime_and_start_extrusion()

        self.log_signal.emit("执行打印轨迹...")
        self.state_signal.emit("打印中...")
        blend = p["blend"]
        pre_stop_mm = p.get("pre_stop", 0)

        if pre_stop_mm > 0 and len(path_3d) > 3:
            cumsum = 0.0
            split_idx = len(path_3d) - 1
            for i in range(len(path_3d) - 1, 0, -1):
                cumsum += float(np.linalg.norm(path_3d[i] - path_3d[i - 1]))
                if cumsum >= pre_stop_mm:
                    split_idx = i
                    break
            split_idx = max(2, split_idx)

            main_path = path_3d[1:split_idx]
            tail_path = path_3d[split_idx:]

            self.log_signal.emit(
                f"预停: 末尾 {pre_stop_mm} mm 提前关电机 "
                f"(前段 {len(main_path)} 点 + 后段 {len(tail_path)} 点)"
            )

            self._move_path(
                main_path,
                blend_radius=blend,
                log_message="机械臂前段轨迹已提交到后台执行",
                stop_motor_on_abort=True,
            )

            self.log_signal.emit("预停: 关闭喷头...")
            self.motor.stop(emergency=False)
            self._abort_if_requested()
            if retract_after:
                self.log_signal.emit(f"回抽 {p['retract']} 脉冲")
                self.motor.backward(p["retract"], speed=50)
                self._sleep_with_abort(0.2)

            self._move_path(
                tail_path,
                blend_radius=0,
                log_message="机械臂后段轨迹已提交到后台执行",
                stop_motor_on_abort=True,
            )
            return

        self._move_path(
            path_3d[1:],
            blend_radius=blend,
            log_message="机械臂轨迹已提交到后台执行",
            stop_motor_on_abort=True,
        )

        if retract_after:
            self._stop_and_retract()
        else:
            self.log_signal.emit("停止喷头...")
            self.motor.stop(emergency=False)

    def _execute_single_trajectory(self, traj):
        self.log_signal.emit(f"✓ {traj}")
        self.state_signal.emit("轨迹已生成")
        path_3d = self._trajectory_to_execution_path(traj)
        self.log_signal.emit(f"路径点数: {len(path_3d)}")
        self._execute_print_path(path_3d, retract_after=True)
        self.log_signal.emit("✓ 打印完成！")

    def _execute_program_step(self, step):
        if step.step_type == STEP_TYPE_TRAVEL:
            self._move_to_point(
                step.params["point_xyz"],
                log_message=f"[{step.name}] 移动到 {step.params['point_xyz']}",
            )
            return

        if step.step_type == STEP_TYPE_PRINT_TRAJECTORY:
            traj = step.params["trajectory"]
            path_3d = self._trajectory_to_execution_path(
                traj,
                default_z=self._resolve_program_step_z(step),
            )
            self.log_signal.emit(f"[{step.name}] 路径点数: {len(path_3d)}")
            self._execute_print_path(path_3d, retract_after=False)
            return

        if step.step_type == STEP_TYPE_LIFT:
            if self._current_point is None:
                raise RuntimeError("lift 步骤之前缺少当前点，无法执行")
            lifted = self._current_point.copy()
            lifted[2] += float(step.params["delta_z"])
            self._move_to_point(
                lifted,
                log_message=f"[{step.name}] 抬升 {step.params['delta_z']} mm 至 {lifted.tolist()}",
            )
            return

        if step.step_type == STEP_TYPE_RETRACT:
            pulses = step.params["pulses"]
            speed = step.params.get("speed", 50)
            self.log_signal.emit(f"[{step.name}] 回抽 {pulses} 脉冲")
            self.motor.backward(pulses, speed=speed)
            self._sleep_with_abort(0.2)
            return

        raise ValueError(f"不支持的执行步骤类型: {step.step_type}")

    def _execute_path_program(self, program):
        self.log_signal.emit(f"✓ PathProgram '{program.name}'，共 {len(program.steps)} 步")
        self.state_signal.emit("轨迹已生成")
        for index, step in enumerate(program.steps, start=1):
            self._abort_if_requested(stop_motor=True)
            self.log_signal.emit(f"执行步骤 {index}/{len(program.steps)}: {step.name} ({step.step_type})")
            self._execute_program_step(step)
        self.log_signal.emit("✓ PathProgram 执行完成！")

    def _run_arm_motion(self, motion_fn, *args, **kwargs):
        result = {"ok": None, "error": None}
        done = threading.Event()

        def _target():
            try:
                result["ok"] = motion_fn(*args, **kwargs)
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        motion_thread = threading.Thread(target=_target, daemon=True)
        motion_thread.start()

        while not done.wait(0.05):
            if self._abort:
                self.log_signal.emit("检测到中止请求，等待机械臂运动退出...")

        if result["error"] is not None:
            raise result["error"]
        if result["ok"] is False:
            raise RuntimeError("机械臂运动执行失败")

    def run(self):
        p = self.params
        arm = self.arm
        motor = self.motor
        try:
            self.log_signal.emit("▶ 开始打印流程...")
            self.state_signal.emit("检查设备...")

            if arm is None:
                raise RuntimeError("机械臂未连接，请先点击“连接机械臂”")
            if motor is None:
                raise RuntimeError("喷头未连接，请先点击“连接喷头”")

            arm.speed = p["arm_speed"]
            arm.acceleration = p["arm_accel"]
            self.log_signal.emit("✓ 复用已连接机械臂")

            try:
                motor.enable()
            except Exception:
                pass
            self.log_signal.emit("✓ 复用已连接喷头")

            self.log_signal.emit("生成轨迹...")
            program = p.get("program")
            if program is not None:
                self._execute_path_program(program)
            else:
                self._execute_single_trajectory(self._build_single_trajectory())
            self.state_signal.emit("完成")
            self.finished_signal.emit(True)

        except InterruptedError:
            self.state_signal.emit("已中止")
            self.finished_signal.emit(False)
        except Exception as e:
            self.log_signal.emit(f"✗ 错误: {e}")
            self.state_signal.emit("出错")
            self.finished_signal.emit(False)

        finally:
            # 线程退出时不主动断开设备，连接生命周期由主窗口管理
            self.arm = None
            self.motor = None


# ==================== 主窗口 ====================

class PrintWindow(QtWidgets.QMainWindow, Ui_PrintWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 设备对象
        self.arm = None
        self.motor = None
        self._traj = None          # 当前生成的轨迹
        self._path_3d = None       # 当前路径点 (N, 3)
        self._print_worker = None
        self._preview_program_builder = None  # 预留: 后续 UI 可切换为复合 PathProgram 预览
        self._execution_program_builder = None  # 预留: 后续 UI 可切换为复合 PathProgram 执行
        self._program_presets = {
            "矩形 + 圆形": self._build_rectangle_circle_program,
        }

        # --- 路径预览图 ---
        self.figure, self.axes = plt.subplots(figsize=(4.5, 6))
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self.widget_plot)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # --- 喷头串口列表 ---
        self._refresh_ports()

        # --- 信号连接 ---
        self.pb_arm_connect.clicked.connect(self._on_arm_connect)
        self.pb_ext_connect.clicked.connect(self._on_ext_connect)
        self.combo_path_mode.currentTextChanged.connect(self._on_path_mode_changed)
        self.combo_program_preset.currentTextChanged.connect(self._on_program_preset_changed)
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self.cb_3d.stateChanged.connect(self._on_3d_changed)
        self.pb_preview.clicked.connect(self._on_preview)
        self.pb_start_print.clicked.connect(self._on_start_print)
        self.pb_abort.clicked.connect(self._on_abort)
        self.pb_ext_start.clicked.connect(self._on_ext_start)
        self.pb_ext_stop.clicked.connect(self._on_ext_stop)
        self.pb_ext_retract.clicked.connect(self._on_ext_retract)
        self.pb_lift.clicked.connect(self._on_lift)
        self.shortcut_abort = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.shortcut_abort.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_abort.activated.connect(self._on_abort)
        self.pb_abort.setToolTip("按 Esc 触发紧急停止")

        # --- 定时刷新 ---
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(200)

        self._on_type_changed(self.combo_type.currentText())
        self._on_3d_changed(self.cb_3d.checkState())
        self._on_path_mode_changed(self.combo_path_mode.currentText())

        # 初始化日志
        self._log("控制台已启动。请配置参数并选择轨迹类型。")
        self._set_print_state("就绪")

    # ── 端口刷新 ─────────────────────────────────────

    def _refresh_ports(self):
        """刷新可用串口列表，默认选中 COM3"""
        self.combo_port.clear()
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            for p in sorted(ports, key=lambda x: x.device):
                self.combo_port.addItem(p.device)
            if self.combo_port.count() == 0:
                self.combo_port.addItem("COM3")
        except Exception:
            self.combo_port.addItems(["COM3", "COM4", "COM5", "COM6"])
        if self.combo_port.findText("COM3") < 0:
            self.combo_port.insertItem(0, "COM3")
        # 默认选中 COM3
        idx = self.combo_port.findText("COM3")
        if idx >= 0:
            self.combo_port.setCurrentIndex(idx)

    # ── 日志 ─────────────────────────────────────────

    def _log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        self.text_log.append(f"[{stamp}] {msg}")

    def _set_lamp(self, lamp, connected):
        color = "#1f9d61" if connected else "#9aa5b1"
        lamp.setStyleSheet(f"background-color:{color};border-radius:7px")

    def _set_print_state(self, state):
        palette = {
            "就绪": ("#e7f7ef", "#147a45"),
            "连接设备...": ("#eaf0ff", "#315fc5"),
            "轨迹已生成": ("#eaf0ff", "#315fc5"),
            "打印中...": ("#fff4db", "#96620d"),
            "完成": ("#e7f7ef", "#147a45"),
            "已中止": ("#fff0e6", "#a54e00"),
            "出错": ("#fdeaea", "#b42323"),
        }
        bg, fg = palette.get(state, ("#eef2f7", "#40516a"))
        self.label_print_state.setText(state)
        self.label_print_state.setStyleSheet(
            f"background:{bg};color:{fg};border-radius:17px;padding:4px 12px;"
        )

    # ── 轨迹类型切换 ───────────────────────────────

    def _on_type_changed(self, t):
        """切换矩形/圆形/弦图/直线参数面板"""
        is_rect = t in ("矩形填充", "矩形轮廓")
        self.group_rect.setVisible(is_rect)
        # 矩形轮廓不需要线宽和起始角（fill-specific）
        is_fill = (t == "矩形填充")
        self.label_lw.setVisible(is_fill)
        self.edit_linewidth.setVisible(is_fill)
        self.label_corner.setVisible(is_fill)
        self.combo_corner.setVisible(is_fill)
        self.group_circle.setVisible(t in ("圆形", "弦图"))
        self.group_line.setVisible(t == "直线")

    def _set_single_trajectory_controls_enabled(self, enabled):
        for widget in (
            self.label_type,
            self.combo_type,
            self.label_center,
            self.edit_cx,
            self.edit_cy,
            self.group_rect,
            self.group_circle,
            self.group_line,
            self.cb_3d,
            self.label_layers,
            self.edit_layers,
            self.label_layer_h,
            self.edit_layer_h,
            self.label_layer_h_unit,
        ):
            widget.setEnabled(enabled)

        self.label_program_preset.setEnabled(not enabled)
        self.combo_program_preset.setEnabled(not enabled)
        self._on_3d_changed(self.cb_3d.checkState())

    def _on_path_mode_changed(self, mode):
        single_mode = (mode == "单条轨迹")
        self._set_single_trajectory_controls_enabled(single_mode)
        if single_mode:
            self._preview_program_builder = None
            self._execution_program_builder = None
            return

        builder = self._get_selected_program_builder()
        self._preview_program_builder = builder
        self._execution_program_builder = builder

    def _on_program_preset_changed(self, _preset_name):
        if self.combo_path_mode.currentText() != "复合程序":
            return
        builder = self._get_selected_program_builder()
        self._preview_program_builder = builder
        self._execution_program_builder = builder

    def _get_selected_program_builder(self):
        return self._program_presets.get(self.combo_program_preset.currentText())

    # ── 3D 复选框切换 ──────────────────────────────

    def _on_3d_changed(self, state):
        single_mode = self.combo_path_mode.currentText() == "单条轨迹"
        enabled = single_mode and state == QtCore.Qt.Checked
        self.edit_layers.setEnabled(enabled)
        self.edit_layer_h.setEnabled(enabled)

    # ── 机械臂连接 ─────────────────────────────────

    def _on_arm_connect(self):
        if self._print_worker is not None and self._print_worker.isRunning():
            self._log("打印进行中，暂不允许断开或重连机械臂")
            return
        if self.arm is not None:
            # 断开
            try:
                self.arm.disconnect()
            except Exception:
                pass
            self.arm = None
            self.pb_arm_connect.setText("连接机械臂")
            self._set_lamp(self.lamp_arm, False)
            self.label_arm_state.setText("未连接")
            self._log("机械臂已断开")
            return

        ip = self.edit_ip.text().strip()
        try:
            speed = _safe_float(self.edit_speed.text(), 10)
            accel = _safe_float(self.edit_accel.text(), 1000)
            self.arm = UR3Controller(robot_ip=ip)
            self.arm.speed = speed
            self.arm.acceleration = accel
            self.pb_arm_connect.setText("断开")
            self._set_lamp(self.lamp_arm, True)
            self.label_arm_state.setText("已连接")
            self._log(f"✓ 机械臂已连接 {ip}, 速度={speed} mm/s, 加速度={accel} mm/s²")
        except Exception as e:
            self._log(f"✗ 机械臂连接失败: {e}")

    # ── 抬高（回到安全高度） ──────────────────────

    def _on_lift(self):
        """移动机械臂到安全位置 (100, 100, 50)"""
        if self.arm is None:
            self._log("✗ 机械臂未连接，无法执行抬高")
            return
        try:
            target = [100.0, 100.0, 50.0]
            self._log(f"⬆ 抬高至安全位置: {target}")
            self.arm.move_to_point(target)
            self._log("✓ 已到达安全位置")
        except Exception as e:
            self._log(f"✗ 抬高失败: {e}")

    # ── 喷头连接 ──────────────────────────────────

    def _on_ext_connect(self):
        if self._print_worker is not None and self._print_worker.isRunning():
            self._log("打印进行中，暂不允许断开或重连喷头")
            return
        if self.motor is not None:
            try:
                self.motor.disable()
                self.motor.close()
            except Exception:
                pass
            self.motor = None
            self.pb_ext_connect.setText("连接喷头")
            self._set_lamp(self.lamp_ext, False)
            self.label_ext_state.setText("未连接")
            self._log("喷头已断开")
            return

        port = self.combo_port.currentText().strip()
        baud = _safe_int(self.combo_baud.currentText(), 115200)
        try:
            self.motor = Motor(serial_port=port, serial_baud=baud)
            self.motor.enable()
            self.pb_ext_connect.setText("断开")
            self._set_lamp(self.lamp_ext, True)
            self.label_ext_state.setText("已连接")
            self._log(f"✓ 喷头已连接 {port} @ {baud}")
        except Exception as e:
            self._log(f"✗ 喷头连接失败: {e}")

    # ── 喷头控制 ──────────────────────────────────

    def _on_ext_start(self):
        if self.motor is None:
            self._log("✗ 喷头未连接")
            return
        rpm = _safe_int(self.edit_ext_rpm.text(), 1)
        self.motor.set_speed(rpm)
        self.motor.move(CW=False)
        self._log(f"喷头启动 {rpm} r/min")

    def _on_ext_stop(self):
        if self.motor is None:
            return
        self.motor.stop()
        self._log("喷头停止")

    def _emergency_stop_devices(self):
        stopped = False

        worker = self._print_worker
        if worker and worker.isRunning():
            worker._abort = True
            if worker.arm:
                try:
                    worker.arm.rtde_c.stopScript()
                    stopped = True
                except Exception:
                    pass
            if worker.motor:
                try:
                    worker.motor.stop(emergency=True)
                    stopped = True
                except Exception:
                    pass

        if self.arm:
            try:
                self.arm.rtde_c.stopScript()
                stopped = True
            except Exception:
                pass

        if self.motor:
            try:
                self.motor.stop(emergency=True)
                stopped = True
            except Exception:
                pass

        return stopped

    def _on_ext_retract(self):
        if self.motor is None:
            self._log("✗ 喷头未连接")
            return
        pulses = _safe_int(self.edit_retract.text(), 2000)
        self.motor.backward(pulses, speed=50)
        self._log(f"回抽 {pulses} 脉冲")

    # ── 预览轨迹 ──────────────────────────────────

    def _on_preview(self):
        try:
            self._traj, self._path_3d = self._build_trajectory()

            for extra_ax in self.figure.axes[1:]:
                extra_ax.remove()
            self.axes.clear()
            if self._path_3d is not None and len(self._path_3d) > 0:
                pts = np.array(self._path_3d)
                if pts.shape[1] >= 3 and len(np.unique(pts[:, 2])) > 1:
                    sc = self.axes.scatter(
                        pts[:, 0], pts[:, 1], c=pts[:, 2], s=9,
                        cmap="viridis", alpha=0.85, label="路径点")
                    self.figure.colorbar(sc, ax=self.axes, fraction=0.035, pad=0.02, label="Z (mm)")
                else:
                    self.axes.plot(pts[:, 0], pts[:, 1], color="#315fc5", linewidth=1.1)
                self.axes.scatter(pts[0, 0], pts[0, 1], c="#1f9d61", s=52, label="起点", zorder=3)
                if len(pts) > 1:
                    self.axes.scatter(pts[-1, 0], pts[-1, 1], c="#d93f3f", s=52, label="终点", zorder=3)
                self.axes.set_aspect("equal")
                self.axes.set_xlabel("X / mm")
                self.axes.set_ylabel("Y / mm")
                self.axes.grid(True, color="#d8e0eb", linewidth=0.6, alpha=0.85)
                self.axes.legend(fontsize=8)
            self.axes.set_title(self._get_preview_title(self._traj), fontsize=11, color="#20324d")
            self.figure.tight_layout(pad=1.2)
            self.canvas.draw()

            total_len = self._get_preview_length(self._traj, self._path_3d)
            n_pts = len(self._path_3d) if self._path_3d is not None else 0
            self.label_traj_info.setText(
                f"轨迹长度: {total_len:.1f} mm  点数: {n_pts}")
            self._log(f"预览: {self._traj}")
        except Exception as e:
            self._log(f"✗ 生成失败: {e}")

    # ── 内部：构建轨迹 ─────────────────────────────

    def _build_trajectory(self):
        """根据当前面板参数构建预览对象和路径点，返回 (traj_or_program, path_3d)"""
        traj_type = self.combo_type.currentText()
        cx = _safe_float(self.edit_cx.text(), 0)
        cy = _safe_float(self.edit_cy.text(), 0)
        is_3d = self.cb_3d.isChecked()
        layers = _safe_int(self.edit_layers.text(), 1) if is_3d else 1
        layer_h = _safe_float(self.edit_layer_h.text(), 1.0)
        work_z = _safe_float(self.edit_workz.text(), 5)

        program = self._build_preview_program()
        if program is not None:
            return program, self._flatten_path_program_points(program, default_z=work_z)

        # corner 映射
        corner_map = {
            "bl": "bottom_left", "br": "bottom_right",
            "tl": "top_left", "tr": "top_right"
        }
        corner = corner_map.get(self.combo_corner.currentText(), "bottom_left")

        if traj_type == "矩形填充":
            w = _safe_float(self.edit_width.text(), 10)
            h = _safe_float(self.edit_height.text(), 10)
            lw = _safe_float(self.edit_linewidth.text(), 1.0)
            traj = TrajectoryFactory.rectangle(
                center=[cx, cy, work_z],
                width=w,
                height=h,
                line_width=lw,
                start_corner=corner,
                name="矩形",
                is_3d=is_3d,
                layers=layers,
                layer_height=layer_h,
            )
        elif traj_type == "矩形轮廓":
            w = _safe_float(self.edit_width.text(), 100)
            h = _safe_float(self.edit_height.text(), 100)
            traj = TrajectoryFactory.rectangle_outline(
                center=[cx, cy, work_z],
                length_x=w,
                width_y=h,
                step_mm=1.0,
                name="矩形轮廓",
                is_3d=is_3d,
                layers=layers,
                layer_height=layer_h,
            )
        elif traj_type == "圆形":
            r = _safe_float(self.edit_radius_c.text(), 90)
            traj = TrajectoryFactory.circle(
                center=[cx, cy], radius=r,
                name="圆形", is_3d=is_3d, layers=layers)
        elif traj_type == "直线":
            x1 = _safe_float(self.edit_line_x1.text(), 90)
            y1 = _safe_float(self.edit_line_y1.text(), 90)
            x2 = _safe_float(self.edit_line_x2.text(), 110)
            y2 = _safe_float(self.edit_line_y2.text(), 110)
            step = _safe_float(self.edit_line_step.text(), 1.0)
            traj = TrajectoryFactory.line(
                start=[x1, y1], end=[x2, y2],
                step_mm=step, name="直线")
        else:  # 弦图
            r = _safe_float(self.edit_radius_c.text(), 90)
            nc = _safe_int(self.edit_nchord.text(), 25)
            traj = TrajectoryFactory.chord_diagram(
                center=[cx, cy], radius=r,
                num_chords=nc,
                name="弦图", is_3d=is_3d, layers=layers)

        return traj, _trajectory_to_path_3d(traj, work_z)

    def _build_preview_program(self):
        """
        预留的复合路径入口。
        当前 UI 尚未接线；后续可将 callable 赋给 self._preview_program_builder。
        """
        if not callable(self._preview_program_builder):
            return None
        program = self._preview_program_builder()
        if program is None:
            return None
        if not isinstance(program, PathProgram):
            raise TypeError("预览程序构建器必须返回 PathProgram")
        return program

    def _build_rectangle_circle_program(self):
        """复合程序示例：矩形 + 圆形。"""
        work_z = _safe_float(self.edit_workz.text(), 5.0)
        safe_z = max(work_z + 1.0, 1.0)
        lift_z = 1.0
        return build_rectangle_circle_program(
            factory=TrajectoryFactory,
            safe_z=safe_z,
            lift_z=lift_z,
        )

    def _build_execution_program(self):
        """
        预留的复合打印入口。
        当前 UI 尚未接线；后续可将 callable 赋给 self._execution_program_builder。
        """
        if not callable(self._execution_program_builder):
            return None
        program = self._execution_program_builder()
        if program is None:
            return None
        if not isinstance(program, PathProgram):
            raise TypeError("执行程序构建器必须返回 PathProgram")
        return program

    def _flatten_path_program_points(self, program, default_z):
        """将 PathProgram 展平成连续的 3D 预览路径。"""
        preview_points = []
        current_point = None

        def _resolve_step_z(step):
            if "work_z" in step.params:
                return float(step.params["work_z"])

            traj = step.params.get("trajectory")
            traj_z = getattr(traj, "z_height", None)
            if traj_z is not None:
                return float(traj_z)

            return float(default_z)

        def _append_point(point_xyz):
            nonlocal current_point
            point = np.array(point_xyz, dtype=float)
            if point.shape[0] < 3:
                point = np.pad(point, (0, 3 - point.shape[0]), constant_values=default_z)
            point = point[:3]
            if current_point is None or not np.allclose(current_point, point):
                preview_points.append(point)
            current_point = point

        for step in program.steps:
            if step.step_type == STEP_TYPE_TRAVEL:
                _append_point(step.params["point_xyz"])
                continue

            if step.step_type == STEP_TYPE_LIFT:
                if current_point is None:
                    raise ValueError("lift 步骤之前缺少当前点，无法生成预览路径")
                lifted = current_point.copy()
                lifted[2] += float(step.params["delta_z"])
                _append_point(lifted)
                continue

            if step.step_type == STEP_TYPE_PRINT_TRAJECTORY:
                traj_points = _trajectory_to_path_3d(
                    step.params["trajectory"],
                    _resolve_step_z(step),
                )
                for point in traj_points:
                    _append_point(point)
                continue

            if step.step_type == STEP_TYPE_RETRACT:
                continue

            raise ValueError(f"不支持的预览步骤类型: {step.step_type}")

        if not preview_points:
            return np.empty((0, 3), dtype=float)
        return np.vstack(preview_points)

    def _get_preview_length(self, traj_or_program, path_3d):
        if hasattr(traj_or_program, "get_total_length"):
            return float(traj_or_program.get_total_length())
        if path_3d is None or len(path_3d) < 2:
            return 0.0
        deltas = np.diff(np.array(path_3d, dtype=float), axis=0)
        return float(np.linalg.norm(deltas, axis=1).sum())

    def _get_preview_title(self, traj_or_program):
        if isinstance(traj_or_program, PathProgram):
            return f"PathProgram '{traj_or_program.name}': {len(traj_or_program.steps)} 步"
        return f"{traj_or_program}"

    # ── 开始打印 ──────────────────────────────────

    def _on_start_print(self):
        if self._print_worker is not None and self._print_worker.isRunning():
            self._log("打印已在运行中")
            return
        if self.arm is None:
            self._log("✗ 机械臂未连接，请先点击“连接机械臂”")
            return
        if self.motor is None:
            self._log("✗ 喷头未连接，请先点击“连接喷头”")
            return

        # 收集参数
        params = {
            "arm_ip": self.edit_ip.text().strip(),
            "arm_speed": _safe_float(self.edit_speed.text(), 10),
            "arm_accel": _safe_float(self.edit_accel.text(), 1000),
            "blend": _safe_float(self.edit_blend.text(), 0.002),
            "ext_port": self.combo_port.currentText().strip(),
            "ext_baud": _safe_int(self.combo_baud.currentText(), 115200),
            "ext_rpm": _safe_int(self.edit_ext_rpm.text(), 1),
            "ext_delay": _safe_float(self.edit_delay.text(), 3),
            "retract": _safe_int(self.edit_retract.text(), 2000),
            "prime": _safe_int(self.edit_prime.text(), 2000),
            "work_z": _safe_float(self.edit_workz.text(), 5),
            "path_type": self.combo_type.currentText(),
            "center_x": _safe_float(self.edit_cx.text(), 100),
            "center_y": _safe_float(self.edit_cy.text(), 100),
            "width": _safe_float(self.edit_width.text(), 10),
            "height": _safe_float(self.edit_height.text(), 10),
            "line_width": _safe_float(self.edit_linewidth.text(), 1.0),
            "radius": _safe_float(self.edit_radius_c.text(), 90),
            "num_chords": _safe_int(self.edit_nchord.text(), 25),
            "line_x1": _safe_float(self.edit_line_x1.text(), 90),
            "line_y1": _safe_float(self.edit_line_y1.text(), 90),
            "line_x2": _safe_float(self.edit_line_x2.text(), 110),
            "line_y2": _safe_float(self.edit_line_y2.text(), 110),
            "line_step": _safe_float(self.edit_line_step.text(), 1.0),
            "is_3d": self.cb_3d.isChecked(),
            "layers": _safe_int(self.edit_layers.text(), 1),
            "layer_height": _safe_float(self.edit_layer_h.text(), 1.0),
            "corner": self.combo_corner.currentText(),
            "pre_stop": _safe_float(self.edit_prestop.text(), 0),
        }
        params["program"] = self._build_execution_program()

        self._log("=" * 40)
        if params["program"] is None:
            self._log(f"轨迹: {params['path_type']}  中心: ({params['center_x']}, {params['center_y']})")
        else:
            self._log(f"程序: {params['program'].name}  步数: {len(params['program'].steps)}")
        self._log(f"速度: {params['arm_speed']} mm/s  喷头: {params['ext_rpm']} r/min")
        if params["is_3d"]:
            self._log(f"3D模式: {params['layers']} 层, 层高 {params['layer_height']} mm")
        if params["pre_stop"] > 0:
            self._log(f"预停距离: {params['pre_stop']} mm（末尾提前关电机）")

        self._print_worker = PrintWorker(params, self.arm, self.motor)
        self._print_worker.log_signal.connect(self._log)
        self._print_worker.state_signal.connect(self._set_print_state)
        self._print_worker.finished_signal.connect(self._on_print_finished)
        self._print_worker.start()

    def _on_print_finished(self, success):
        if success:
            self._log("=" * 40)
        else:
            self._log("打印异常终止")
        self._set_print_state("就绪")

    def _on_abort(self):
        if self._emergency_stop_devices():
            self._log("⚠ 紧急停止！(Esc)")
            self._set_print_state("已中止")
        else:
            self._log("Esc 已按下，但当前没有可停止的设备")

    # ── 定时状态刷新 ──────────────────────────────

    def _refresh_status(self):
        if self.arm:
            try:
                tcp = self.arm.get_tcp()
                self.label_tcp.setText(
                    f"TCP: X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}")
            except Exception:
                pass

        motor_poll_blocked = (
            self._print_worker is not None
            and self._print_worker.isRunning()
        )

        if self.motor and not motor_poll_blocked:
            try:
                pos = self.motor.get_absolute_position()
                spd = self.motor.get_current_speed()
                self.label_ext_pos.setText(
                    f"位置: {pos} Pls  速度: {spd} r/min")
            except Exception:
                pass

    # ── 关闭事件 ──────────────────────────────────

    def closeEvent(self, event):
        if self._print_worker and self._print_worker.isRunning():
            self._emergency_stop_devices()
            self._print_worker.wait(3000)
        if self.motor:
            try:
                self.motor.stop(emergency=False)
                self.motor.disable()
                self.motor.close()
            except Exception:
                pass
        if self.arm:
            try:
                self.arm.disconnect()
            except Exception:
                pass
        self.timer.stop()
        event.accept()


# ==================== 启动入口 ====================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PrintWindow()
    window.show()
    sys.exit(app.exec_())
