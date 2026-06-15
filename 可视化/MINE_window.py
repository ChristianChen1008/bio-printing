# -*- coding: utf-8 -*-
"""
3D打印综合控制台 — 窗口逻辑
整合机械臂、喷头电机、轨迹生成，提供完整的打印控制界面。
"""

import sys
import os
import time
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

from 完整代码.UR3Controller import UR3Controller
from 完整代码.Motor import Motor
from 完整代码.Trajectory import TrajectoryFactory


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


# ==================== 打印工作线程 ====================

class PrintWorker(QThread):
    """后台执行打印流程，避免阻塞 UI"""
    log_signal = pyqtSignal(str)          # 日志消息
    state_signal = pyqtSignal(str)        # 状态变化
    finished_signal = pyqtSignal(bool)    # 完成信号 (True=成功, False=失败或中止)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._abort = False
        self.arm = None    # 暴露给主线程做紧急停止
        self.motor = None

    def run(self):
        p = self.params
        arm = None
        motor = None
        try:
            self.log_signal.emit("▶ 开始打印流程...")
            self.state_signal.emit("连接设备...")

            # 1. 连接机械臂
            self.log_signal.emit(f"连接机械臂 {p['arm_ip']} ...")
            arm = UR3Controller(robot_ip=p["arm_ip"])
            self.arm = arm
            arm.speed = p["arm_speed"]
            arm.acceleration = p["arm_accel"]
            self.log_signal.emit("✓ 机械臂已连接")

            # 2. 连接喷头电机
            self.log_signal.emit(f"连接喷头 {p['ext_port']} @ {p['ext_baud']} ...")
            motor = Motor(serial_port=p["ext_port"], serial_baud=p["ext_baud"])
            self.motor = motor
            motor.enable()
            self.log_signal.emit("✓ 喷头电机已连接并上电")

            # 3. 生成轨迹
            self.log_signal.emit("生成轨迹...")
            traj_type = p["path_type"]
            cx, cy = p["center_x"], p["center_y"]
            is_3d = p["is_3d"]
            layers = p["layers"] if is_3d else 1

            if traj_type == "矩形填充":
                traj = TrajectoryFactory.rectangle(
                    center=[cx, cy, p["work_z"]],
                    width=p["width"],
                    height=p["height"],
                    line_width=p["line_width"],
                    start_corner=p.get("corner", "bottom_left"),
                    name="矩形",
                    is_3d=is_3d,
                    layers=layers,
                    layer_height=p["layer_height"],
                )
            elif traj_type == "矩形轮廓":
                traj = TrajectoryFactory.rectangle_outline(
                    center=[cx, cy, p["work_z"]],
                    length_x=p["width"],
                    width_y=p["height"],
                    step_mm=1.0,
                    name="矩形轮廓",
                    is_3d=is_3d,
                    layers=layers,
                    layer_height=p["layer_height"],
                )
            elif traj_type == "圆形":
                traj = TrajectoryFactory.circle(
                    center=[cx, cy],
                    radius=p["radius"],
                    name="圆形",
                    is_3d=is_3d,
                    layers=layers,
                )
            elif traj_type == "弦图":
                traj = TrajectoryFactory.chord_diagram(
                    center=[cx, cy],
                    radius=p["radius"],
                    num_chords=p["num_chords"],
                    name="弦图",
                    is_3d=is_3d,
                    layers=layers,
                )
            elif traj_type == "直线":
                traj = TrajectoryFactory.line(
                    start=[p["line_x1"], p["line_y1"]],
                    end=[p["line_x2"], p["line_y2"]],
                    step_mm=p["line_step"],
                    name="直线",
                )
            else:
                raise ValueError(f"未知轨迹类型: {traj_type}")

            self.log_signal.emit(f"✓ {traj}")
            self.state_signal.emit("轨迹已生成")

            # 4. 提取路径点
            if traj.generated_points is not None:
                raw = np.array(traj.generated_points, dtype=float)
                if raw.ndim == 2 and raw.shape[1] >= 3:
                    path_3d = raw[:, :3]
                else:
                    z = p["work_z"]
                    path_3d = np.column_stack([raw, np.full(len(raw), z)])
            else:
                path_2d = traj.generate_path_points(step_mm=5.0)
                z = p["work_z"]
                path_3d = np.column_stack([path_2d, np.full(len(path_2d), z)])

            self.log_signal.emit(f"路径点数: {len(path_3d)}")

            if len(path_3d) < 2:
                raise RuntimeError("路径点不足，无法执行")

            # 5. 移动到起点
            self.log_signal.emit(f"移动到起点: {path_3d[0]}")
            arm.move_to_point(path_3d[0])
            time.sleep(0.5)

            if self._abort:
                self.state_signal.emit("已中止")
                self.finished_signal.emit(False)
                return

            # 6a. 预挤出（脉冲模式，走完自动停）
            prime_pulses = p["prime"]
            if prime_pulses > 0:
                self.log_signal.emit(f"预挤出 {prime_pulses} 脉冲...")
                motor.set_speed(p["ext_rpm"])
                motor.forward(prime_pulses)
                if self._abort:
                    self.state_signal.emit("已中止")
                    self.finished_signal.emit(False)
                    return

            # 6b. 持续挤出 + 等待出料稳定（可设0跳过）
            ext_rpm = p["ext_rpm" ]
            delay = p["ext_delay"]
            if ext_rpm > 0 or delay > 0:
                self.log_signal.emit(f"持续挤出 {ext_rpm} r/min, 等待 {delay} 秒")
                motor.set_speed(ext_rpm)
                motor.move(CW=False)
                if delay > 0:
                    time.sleep(delay)
            else:
                self.log_signal.emit("跳过持续挤出（速度=0 且 延迟=0）")

            if self._abort:
                motor.stop()
                self.state_signal.emit("已中止")
                self.finished_signal.emit(False)
                return

            # 7. 执行轨迹（支持预停距离：末尾前N mm提前关电机）
            self.log_signal.emit("执行打印轨迹...")
            self.state_signal.emit("打印中...")
            blend = p["blend"]
            pre_stop_mm = p.get("pre_stop", 0)

            if pre_stop_mm > 0 and len(path_3d) > 3:
                # 从末尾往前计算累积距离，找到拆分点
                cumsum = 0.0
                split_idx = len(path_3d) - 1
                for i in range(len(path_3d) - 1, 0, -1):
                    cumsum += float(np.linalg.norm(path_3d[i] - path_3d[i - 1]))
                    if cumsum >= pre_stop_mm:
                        split_idx = i
                        break
                split_idx = max(2, split_idx)  # 至少保留起点+1点给前段

                main_path = path_3d[1:split_idx]   # 前段：电机运行
                tail_path = path_3d[split_idx:]     # 后段：靠余压

                self.log_signal.emit(
                    f"预停: 末尾 {pre_stop_mm} mm 提前关电机 "
                    f"(前段 {len(main_path)} 点 + 后段 {len(tail_path)} 点)"
                )

                # 走前段（电机开着）
                arm.move_path(main_path, blend_radius=blend)

                if self._abort:
                    motor.stop()
                    self.state_signal.emit("已中止")
                    self.finished_signal.emit(False)
                    return

                # 提前关电机 + 回抽
                self.log_signal.emit("预停: 关闭喷头...")
                motor.stop(emergency=False)
                retract_pulses = p["retract"]
                self.log_signal.emit(f"回抽 {retract_pulses} 脉冲")
                motor.backward(retract_pulses, speed=50)
                time.sleep(0.2)

                # 走后段（靠余压挤出，不用 blend 避免平滑到路径外）
                arm.move_path(tail_path, blend_radius=0)

                self.log_signal.emit("✓ 打印完成！（预停模式）")
                self.state_signal.emit("完成")
                self.finished_signal.emit(True)
                return
            else:
                # 无预停：原始行为
                arm.move_path(path_3d[1:], blend_radius=blend)

            # 8. 停止喷头 + 回抽
            self.log_signal.emit("停止喷头...")
            motor.stop(emergency=False)
            retract_pulses = p["retract"]
            self.log_signal.emit(f"回抽 {retract_pulses} 脉冲")
            motor.backward(retract_pulses, speed=50)
            time.sleep(0.3)

            self.log_signal.emit("✓ 打印完成！")
            self.state_signal.emit("完成")
            self.finished_signal.emit(True)

        except Exception as e:
            self.log_signal.emit(f"✗ 错误: {e}")
            self.state_signal.emit("出错")
            self.finished_signal.emit(False)

        finally:
            # 清理
            self.arm = None
            self.motor = None
            if motor:
                try:
                    motor.stop(emergency=False)
                    motor.disable()
                    motor.close()
                    self.log_signal.emit("喷头已断开")
                except Exception:
                    pass
            if arm:
                try:
                    arm.disconnect()
                    self.log_signal.emit("机械臂已断开")
                except Exception:
                    pass


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
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self.cb_3d.stateChanged.connect(self._on_3d_changed)
        self.pb_preview.clicked.connect(self._on_preview)
        self.pb_start_print.clicked.connect(self._on_start_print)
        self.pb_abort.clicked.connect(self._on_abort)
        self.pb_ext_start.clicked.connect(self._on_ext_start)
        self.pb_ext_stop.clicked.connect(self._on_ext_stop)
        self.pb_ext_retract.clicked.connect(self._on_ext_retract)
        self.pb_lift.clicked.connect(self._on_lift)

        # --- 定时刷新 ---
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(200)

        # 初始化日志
        self._log("控制台已启动。请配置参数并选择轨迹类型。")
        self._set_print_state("就绪")

    # ── 端口刷新 ─────────────────────────────────────

    def _refresh_ports(self):
        """刷新可用串口列表，默认选中 COM5"""
        self.combo_port.clear()
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            for p in sorted(ports, key=lambda x: x.device):
                self.combo_port.addItem(p.device)
            if self.combo_port.count() == 0:
                self.combo_port.addItem("COM5")
        except Exception:
            self.combo_port.addItems(["COM3", "COM4", "COM5", "COM6"])
        # 默认选中 COM5
        idx = self.combo_port.findText("COM5")
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

    # ── 3D 复选框切换 ──────────────────────────────

    def _on_3d_changed(self, state):
        enabled = state == QtCore.Qt.Checked
        self.edit_layers.setEnabled(enabled)
        self.edit_layer_h.setEnabled(enabled)

    # ── 机械臂连接 ─────────────────────────────────

    def _on_arm_connect(self):
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
            self.axes.set_title(f"{self._traj}", fontsize=11, color="#20324d")
            self.figure.tight_layout(pad=1.2)
            self.canvas.draw()

            total_len = self._traj.get_total_length()
            n_pts = len(self._path_3d) if self._path_3d is not None else 0
            self.label_traj_info.setText(
                f"轨迹长度: {total_len:.1f} mm  点数: {n_pts}")
            self._log(f"预览: {self._traj}")
        except Exception as e:
            self._log(f"✗ 生成失败: {e}")

    # ── 内部：构建轨迹 ─────────────────────────────

    def _build_trajectory(self):
        """根据当前面板参数构建轨迹和路径点，返回 (traj, path_3d)"""
        traj_type = self.combo_type.currentText()
        cx = _safe_float(self.edit_cx.text(), 0)
        cy = _safe_float(self.edit_cy.text(), 0)
        is_3d = self.cb_3d.isChecked()
        layers = _safe_int(self.edit_layers.text(), 1) if is_3d else 1
        layer_h = _safe_float(self.edit_layer_h.text(), 1.0)
        work_z = _safe_float(self.edit_workz.text(), 5)

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

        # 提取路径点
        if traj.generated_points is not None:
            raw = np.array(traj.generated_points, dtype=float)
            if raw.ndim == 2 and raw.shape[1] >= 3:
                path_3d = raw[:, :3]
            else:
                path_3d = np.column_stack([raw, np.full(len(raw), work_z)])
        else:
            path_2d = traj.generate_path_points(step_mm=5.0)
            path_3d = np.column_stack([path_2d, np.full(len(path_2d), work_z)])

        return traj, path_3d

    # ── 开始打印 ──────────────────────────────────

    def _on_start_print(self):
        if self._print_worker is not None and self._print_worker.isRunning():
            self._log("打印已在运行中")
            return

        # 先断开主窗口已有的连接，避免串口/IP冲突（worker会自行连接）
        if self.motor is not None:
            try:
                self.motor.stop(emergency=False)
                self.motor.disable()
                self.motor.close()
            except Exception:
                pass
            self.motor = None
            self.pb_ext_connect.setText("连接喷头")
            self._set_lamp(self.lamp_ext, False)
            self.label_ext_state.setText("未连接")
            self._log("已释放主窗口喷头连接")

        if self.arm is not None:
            try:
                self.arm.disconnect()
            except Exception:
                pass
            self.arm = None
            self.pb_arm_connect.setText("连接机械臂")
            self._set_lamp(self.lamp_arm, False)
            self.label_arm_state.setText("未连接")
            self._log("已释放主窗口机械臂连接")

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

        self._log("=" * 40)
        self._log(f"轨迹: {params['path_type']}  中心: ({params['center_x']}, {params['center_y']})")
        self._log(f"速度: {params['arm_speed']} mm/s  喷头: {params['ext_rpm']} r/min")
        if params["is_3d"]:
            self._log(f"3D模式: {params['layers']} 层, 层高 {params['layer_height']} mm")
        if params["pre_stop"] > 0:
            self._log(f"预停距离: {params['pre_stop']} mm（末尾提前关电机）")

        self._print_worker = PrintWorker(params)
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
        if self._print_worker and self._print_worker.isRunning():
            self._print_worker._abort = True
            self._log("⚠ 紧急停止！")
            self._set_print_state("已中止")
            w = self._print_worker
            # 直接停止 worker 内部连接的机械臂和电机
            if w.arm:
                try:
                    w.arm.rtde_c.stopScript()
                except Exception:
                    pass
            if w.motor:
                try:
                    w.motor.stop(emergency=True)
                except Exception:
                    pass

    # ── 定时状态刷新 ──────────────────────────────

    def _refresh_status(self):
        if self.arm:
            try:
                tcp = self.arm.get_tcp()
                self.label_tcp.setText(
                    f"TCP: X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}")
            except Exception:
                pass

        if self.motor:
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
            self._print_worker._abort = True
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
