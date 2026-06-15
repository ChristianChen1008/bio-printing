"""
3D打印主程序 —— 逐层挤出控制（优化版）
============================================

核心优化：
  每一层独立执行以下循环：
    预挤出(Prime) → 持续挤出+打印 → 回抽(Retract) → 停电机 →
    抬升Z轴（在当前位置垂直抬升，避免刮伤已打印层）→
    水平回退到原点（在安全高度移动，电机不运行）→ 下一层

解决问题：
  - 层间过渡时先抬Z再水平移动，防止喷嘴刮伤/破坏已打印层的图形
  - 层间过渡时喷嘴漏料导致下一层起始段断料
  - 回抽后的残余压力在过渡移动中释放完毕，下一层开始前重新预挤出补偿

适用场景：
  - 矩形填充面 (rectangle)
  - 矩形轮廓 (rectangle_outline)
  - 可扩展至其他轨迹类型

用法：
  1. 修改下方 ===== 配置参数 ===== 区块
  2. 运行：python 主程序（3D打印-层优化）.py
"""

import time
import sys
import numpy as np

from Motor import Motor
from Trajectory import TrajectoryFactory
from UR3Controller import UR3Controller


# ==================== 配置参数 ====================

# ---- 硬件连接 ----
ROBOT_IP        = "169.254.45.1"   # UR3机械臂IP
SERIAL_PORT     = "COM3"           # 挤出电机串口
BAUDRATE        = 115200           # 串口波特率

# ---- 打印几何形状（矩形填充面）----
RECT_CENTER     = [100, 100, 0]    # 矩形中心 [x, y, z_base]（z_base 为第一层基准高度）
RECT_WIDTH      = 10               # 矩形宽度 mm（X方向）
RECT_LENGTH     = 10               # 矩形长度 mm（Y方向）
LINE_WIDTH      = 1.0              # 填充线宽 mm（蛇形线间距）
START_CORNER    = "bottom_left"    # 起始角：bottom_left / bottom_right / top_left / top_right
START_AXIS      = "y"              # 蛇形扫描方向："y"=沿Y轴来回，"x"=沿X轴来回
LAYERS          = 5                # 总打印层数
LAYER_HEIGHT    = 1.0              # 每层抬升高度 mm

# ---- 轨迹类型选择 ----
# "rectangle" = 矩形填充面（蛇形）
# "rectangle_outline" = 矩形轮廓（仅边框）
TRAJECTORY_TYPE = "rectangle"

# ---- 机械臂运动参数 ----
ARM_SPEED           = 300           # 打印时运动速度 mm/s
ARM_ACCEL           = 1000          # 加速度 mm/s²
BLEND_RADIUS        = 0.002         # 路径混合半径 m
TRANSITION_SPEED    = 500           # 层间过渡运动速度 mm/s（可设比打印速度更快）
TRANSITION_ACCEL    = 1500          # 层间过渡加速度 mm/s²

# ---- 挤出电机参数 ----
EXTRUSION_SPEED     = 30            # 挤出速度 r/min
RETRACT_SPEED       = 50            # 回抽速度 r/min
PRIME_PULSES        = 200           # 每层开始前预挤出脉冲数（补偿过渡期间漏料）
RETRACT_PULSES      = 300           # 每层结束后回抽脉冲数（防止层间漏料）
EXTRUDE_DELAY       = 1.0           # 预挤出后等待出料稳定的时间(秒)，设0跳过

# ---- 安全与调试 ----
DRY_RUN             = False         # True=仅模拟运行（不连接硬件，打印路径信息）
PAUSE_BETWEEN_LAYERS = False        # True=每层结束后暂停等待用户确认
STEP_MODE            = False        # True=逐点打印（调试用）

# =====================================================


def connect_hardware():
    """连接机械臂和挤出电机，返回 (arm, motor)。失败时退出程序。"""
    arm, motor = None, None

    print("初始化设备...")

    # 1. 连接机械臂
    try:
        arm = UR3Controller(robot_ip=ROBOT_IP)
        arm.speed        = ARM_SPEED
        arm.acceleration = ARM_ACCEL
        print(f"  ✓ 机械臂已连接: {ROBOT_IP}")
    except Exception as e:
        print(f"  ✗ 机械臂连接失败: {e}")
        sys.exit(1)

    # 2. 连接挤出电机
    try:
        motor = Motor(serial_port=SERIAL_PORT, serial_baud=BAUDRATE)
        motor.enable()
        print(f"  ✓ 挤出电机已连接: {SERIAL_PORT}")
    except Exception as e:
        print(f"  ✗ 电机连接失败: {e}")
        arm.disconnect()
        sys.exit(1)

    return arm, motor


def disconnect_hardware(arm, motor):
    """安全断开所有硬件连接。"""
    if motor is not None:
        try:
            motor.stop(emergency=False)
            time.sleep(0.2)
            motor.close()
            print("  ✓ 电机已断开")
        except Exception as e:
            print(f"  ⚠ 电机断开异常: {e}")

    if arm is not None:
        try:
            arm.disconnect()
            print("  ✓ 机械臂已断开")
        except Exception as e:
            print(f"  ⚠ 机械臂断开异常: {e}")


def generate_trajectory():
    """根据配置生成轨迹对象。"""
    if TRAJECTORY_TYPE == "rectangle":
        traj = TrajectoryFactory.rectangle(
            center=RECT_CENTER,
            width=RECT_WIDTH,
            height=RECT_LENGTH,
            line_width=LINE_WIDTH,
            start_corner=START_CORNER,
            start_axis=START_AXIS,
            name="矩形填充-3D打印",
            is_3d=True,
            layers=LAYERS,
            layer_height=LAYER_HEIGHT,
        )
    elif TRAJECTORY_TYPE == "rectangle_outline":
        traj = TrajectoryFactory.rectangle_outline(
            center=RECT_CENTER,
            length_x=RECT_WIDTH,
            width_y=RECT_LENGTH,
            name="矩形轮廓-3D打印",
            is_3d=True,
            layers=LAYERS,
            layer_height=LAYER_HEIGHT,
        )
    else:
        raise ValueError(f"不支持的轨迹类型: {TRAJECTORY_TYPE}")

    print(traj)
    print(f"  总路径点数: {len(traj.generated_points)}")
    return traj


def pre_extrude(motor):
    """预挤出：补偿过渡期间漏掉的料，确保下一层起始段不缺料。"""
    if PRIME_PULSES > 0:
        print(f"    预挤出 {PRIME_PULSES} 脉冲...")
        motor.set_speed(EXTRUSION_SPEED)
        motor.forward(PRIME_PULSES)
        time.sleep(0.1)

    if EXTRUDE_DELAY > 0:
        print(f"    等待出料稳定 {EXTRUDE_DELAY}s...")
        motor.set_speed(EXTRUSION_SPEED)
        motor.move(CW=False)
        time.sleep(EXTRUDE_DELAY)


def retract(motor):
    """回抽：层结束时回抽材料，防止层间过渡时漏料。"""
    if RETRACT_PULSES > 0:
        print(f"    回抽 {RETRACT_PULSES} 脉冲...")
        motor.set_speed(RETRACT_SPEED)
        motor.backward(RETRACT_PULSES)
        time.sleep(0.1)


def execute_layer_print(arm, motor, print_pts):
    """
    执行单层打印：遍历打印路径点，挤出电机持续运行。
    打印点至少包含起点（已通过 move_to_point 到达）和后续路径点。
    """
    if len(print_pts) <= 1:
        print("    ⚠ 该层路径点不足，跳过")
        return

    # 启动挤出电机
    motor.set_speed(EXTRUSION_SPEED)
    motor.move(CW=False)

    if STEP_MODE:
        # 逐点模式（调试用）：每个点单独移动
        for i, pt in enumerate(print_pts[1:], start=1):
            print(f"      点 {i}/{len(print_pts)-1}: {pt}")
            arm.move_to_point(pt)
    else:
        # 正常模式：连续路径
        arm.move_path(print_pts[1:], blend_radius=BLEND_RADIUS)

    # 停止挤出
    motor.stop(emergency=False)
    time.sleep(0.1)


def execute_transition(arm, trans_pts):
    """
    执行层间过渡：仅机械臂运动，挤出电机不运行。
    过渡段包括：回退到原点（水平）+ 抬升Z轴（垂直）。
    """
    if not trans_pts or len(trans_pts) == 0:
        print("    （无过渡段，已完成最后一层）")
        return

    print(f"    过渡段 {len(trans_pts)} 个路径点，速度={TRANSITION_SPEED}mm/s")

    # 临时切换到过渡速度
    orig_speed = arm.speed
    orig_accel = arm.acceleration
    arm.speed        = TRANSITION_SPEED
    arm.acceleration = TRANSITION_ACCEL

    try:
        arm.move_path(trans_pts, blend_radius=BLEND_RADIUS)
    finally:
        # 恢复打印速度
        arm.speed        = orig_speed
        arm.acceleration = orig_accel


def print_layer_summary(layer_groups):
    """打印层分组摘要信息。"""
    print("\n" + "=" * 60)
    print("层分组信息:")
    print("-" * 60)
    for g in layer_groups:
        n_print = len(g['print_pts'])
        n_trans = len(g['transition_pts'])
        print(f"  第 {g['layer']+1} 层: "
              f"Z={g['z']:.1f}mm, "
              f"打印点={n_print}, "
              f"过渡点={n_trans}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("3D打印 —— 逐层挤出控制（优化版）")
    print(f"轨迹类型: {TRAJECTORY_TYPE}")
    print(f"层数: {LAYERS}  |  层高: {LAYER_HEIGHT} mm")
    print(f"打印速度: {ARM_SPEED} mm/s  |  挤出速度: {EXTRUSION_SPEED} r/min")
    print(f"预挤出: {PRIME_PULSES} 脉冲  |  回抽: {RETRACT_PULSES} 脉冲")
    print(f"过渡速度: {TRANSITION_SPEED} mm/s")
    if DRY_RUN:
        print("*** 干运行模式：不连接硬件 ***")
    print("=" * 60)

    # ========== 生成轨迹 ==========
    print("\n[1/4] 生成轨迹...")
    traj = generate_trajectory()
    layer_groups = traj.get_layer_groups()
    print_layer_summary(layer_groups)

    if DRY_RUN:
        print("\n干运行完成，退出。")
        return

    # ========== 连接硬件 ==========
    print("\n[2/4] 连接硬件...")
    arm, motor = connect_hardware()

    try:
        # ========== 逐层打印 ==========
        print(f"\n[3/4] 开始逐层打印（共 {len(layer_groups)} 层）...")

        for group in layer_groups:
            layer_num = group['layer'] + 1  # 人类可读层号（1-based）
            z         = group['z']
            print_pts = group['print_pts']
            trans_pts = group['transition_pts']

            print(f"\n{'─' * 50}")
            print(f"第 {layer_num}/{len(layer_groups)} 层  |  Z={z:.1f} mm")
            print(f"  打印段: {len(print_pts)} 点  |  过渡段: {len(trans_pts)} 点")

            # ---- 步骤A: 移动到该层起点（不挤出）----
            first_pt = print_pts[0]
            print(f"  ① 移动到层起点: {[f'{v:.1f}' for v in first_pt]}")
            arm.speed        = TRANSITION_SPEED
            arm.acceleration = TRANSITION_ACCEL
            arm.move_to_point(first_pt)
            arm.speed        = ARM_SPEED
            arm.acceleration = ARM_ACCEL
            time.sleep(0.3)

            # ---- 步骤B: 预挤出（补偿漏料）----
            print(f"  ② 预挤出")
            pre_extrude(motor)

            # ---- 步骤C: 持续挤出 + 打印 ----
            print(f"  ③ 打印中...")
            execute_layer_print(arm, motor, print_pts)

            # ---- 步骤D: 回抽 ----
            print(f"  ④ 回抽")
            retract(motor)

            # ---- 步骤E: 过渡（先抬Z再回原点，仅机械臂，电机停）----
            if trans_pts:
                print(f"  ⑤ 层间过渡: 抬升Z轴 → 水平回原点（电机停止，仅机械臂运动）...")
                execute_transition(arm, trans_pts)
            else:
                print(f"  ⑤ 最后一层，无过渡段")

            print(f"  ✓ 第 {layer_num} 层完成")

            # 层间暂停（调试/观察用）
            if PAUSE_BETWEEN_LAYERS and layer_num < len(layer_groups):
                input(f"\n  按 Enter 继续下一层...")

        print(f"\n{'─' * 50}")
        print("✓ 全部层打印完成!")

    except KeyboardInterrupt:
        print("\n⚠ 用户中断")
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ========== 清理 ==========
        print("\n[4/4] 清理与断开...")
        print("  停止挤出电机...")
        if motor is not None:
            try:
                motor.stop(emergency=False)
                time.sleep(0.2)
            except Exception:
                pass
        disconnect_hardware(arm, motor)
        print("程序结束。")


if __name__ == "__main__":
    main()
