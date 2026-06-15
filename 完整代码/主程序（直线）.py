import time
import sys
import numpy as np

from Motor import Motor
from Trajectory import TrajectoryFactory
from UR3Controller import UR3Controller

# ==================== 配置参数 ====================
ROBOT_IP        = "169.254.45.1"
SERIAL_PORT     = "COM5"
BAUDRATE        = 115200

# 直线参数
LINE_START      = [10, 10]     # 起点 [x, y]，单位mm
LINE_END        = [200,200]   # 终点 [x, y]，单位mm
LINE_STEP       = 1.0          # 采样步长，单位mm

# 机械臂运动参数
ARM_SPEED       = 300            # mm/s  （可自行修改）
ARM_ACCEL       = 1000         # mm/s²
BLEND_RADIUS    = 0.002        # 路径混合半径，单位m

# 喷头挤出参数
EXTRUSION_SPEED = 30            # r/min  （可自行修改）
EXTRUDE_DELAY   = 10           # 启动喷头后等待出料稳定的时间(s)，设为0跳过
PRIME_PULSES    = 0            # 预挤出脉冲数，设为0跳过

# 工作高度
WORK_Z          = 5.0          # mm，机械臂Z轴工作高度

# =====================================================

def main():
    print("=" * 60)
    print("直线轨迹速度测试")
    print(f"起点: {LINE_START}  终点: {LINE_END}")
    print(f"机械臂速度: {ARM_SPEED} mm/s  挤出速度: {EXTRUSION_SPEED} r/min")
    print("=" * 60)
    print("初始化设备...")

    # 1. 连接机械臂
    try:
        arm = UR3Controller(robot_ip=ROBOT_IP)
    except Exception as e:
        print(f"机械臂连接失败: {e}")
        sys.exit(1)

    arm.speed        = ARM_SPEED
    arm.acceleration = ARM_ACCEL

    # 2. 连接喷头电机
    try:
        motor = Motor(serial_port=SERIAL_PORT, serial_baud=BAUDRATE)
    except Exception as e:
        print(f"电机连接失败: {e}")
        arm.disconnect()
        sys.exit(1)

    motor.enable()

    try:
        # 3. 生成直线轨迹
        traj = TrajectoryFactory.line(
            start=LINE_START,
            end=LINE_END,
            step_mm=LINE_STEP,
            name="直线速度测试",
        )
        print(traj)

        all_points = np.array(traj.generated_points)
        print(f"路径点数: {len(all_points)}")

        # 将2D点补上工作高度Z，构成3D路径
        path_3d = np.column_stack([all_points, np.full(len(all_points), WORK_Z)])

        # 4. 先移动到起点（不挤出）
        print(f"\n移动到起点: {path_3d[0]}")
        arm.move_to_point(path_3d[0])
        time.sleep(0.5)

        # 5a. 预挤出（脉冲模式，走完自动停）
        if PRIME_PULSES > 0:
            print(f"预挤出 {PRIME_PULSES} 脉冲...")
            motor.set_speed(EXTRUSION_SPEED)
            motor.forward(PRIME_PULSES)

        # 5b. 持续挤出 + 等待出料稳定（可设0跳过）
        if EXTRUSION_SPEED > 0 or EXTRUDE_DELAY > 0:
            print(f"持续挤出 {EXTRUSION_SPEED} r/min, 等待 {EXTRUDE_DELAY} 秒")
            motor.set_speed(EXTRUSION_SPEED)
            motor.move(CW=False)
            if EXTRUDE_DELAY > 0:
                time.sleep(EXTRUDE_DELAY)
        else:
            print("跳过持续挤出（速度=0 且 延迟=0）")

        # 6. 执行直线轨迹
        print("开始打印直线...")
        arm.move_path(path_3d[1:], blend_radius=BLEND_RADIUS)
        print("轨迹执行完成")

    except Exception as e:
        print(f"运行出错: {e}")

    finally:
        # 7. 无论是否出错，都停止喷头并断开设备
        print("停止喷头...")
        motor.stop(emergency=False)
        time.sleep(0.2)

        motor.close()
        arm.disconnect()
        print("程序结束。")


if __name__ == "__main__":
    main()
