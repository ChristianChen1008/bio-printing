import serial
import struct
import time
import numpy as np


def crc16(data):
    """计算CRC16校验码"""
    crc = 0xffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xa001
            else:
                crc >>= 1
    return crc


def write_limit():
    motor_lim_max = motor.get_absolute_position()
    motor_lim_min = motor_lim_max - 120000
    lim = [motor_lim_min]
    np.save('motor_lim.npy', lim)



class Motor_Serial:
    """底层串口通信类，负责Modbus RTU协议的收发"""

    def __init__(self, serial_port, serial_baud=115200, timeout=2):
        self.ser = serial.Serial(serial_port, serial_baud, timeout=timeout)
        self.salve = 1
        self.read_single_code   = 0x03
        self.read_multiple_code = 0x04
        self.write_single_code  = 0x06
        self.write_multiple_code = 0x10

    def write_single(self, address, value):
        """写单个WORD寄存器（0x06）"""
        function_code = self.write_single_code
        if value < 0:
            value = (1 << 16) + value
        request_frame = struct.pack('>BBHH',
                                    self.salve,
                                    function_code,
                                    address,
                                    value)
        crc = crc16(request_frame)
        request_frame += struct.pack('<H', crc)
        self.ser.write(request_frame)
        self.ser.read(8)

    def read_single(self, address):
        """读单个WORD寄存器（0x03），返回无符号16位整数"""
        function_code = self.read_single_code
        register_num = 1
        request_frame = struct.pack('>BBHH',
                                    self.salve,
                                    function_code,
                                    address,
                                    register_num)
        crc = crc16(request_frame)
        request_frame += struct.pack('<H', crc)
        self.ser.write(request_frame)
        response = self.ser.read(7)
        time.sleep(0.01)
        if len(response) == 7:
            _, _, num_bytes, value = struct.unpack('>BBBH', response[:5])
            if num_bytes == 2:
                return value
        return None

    def write_multiple(self, address, value):
        """写双WORD寄存器（0x10），用于DWORD（32位）"""
        function_code = self.write_multiple_code
        register_num = 2
        bytes_num = 4
        if value < 0:
            value = (1 << 32) + value
        value_high = value >> 16
        value_low  = value & 0xffff
        request_frame = struct.pack('>BBHHBHH',
                                    self.salve,
                                    function_code,
                                    address,
                                    register_num,
                                    bytes_num,
                                    value_low,
                                    value_high)
        crc = crc16(request_frame)
        request_frame += struct.pack('<H', crc)
        self.ser.write(request_frame)
        self.ser.read(8)
        time.sleep(0.01)

    def read_multiple(self, address):
        """读双WORD寄存器（0x03），用于DWORD（32位），返回无符号32位整数"""
        function_code = self.read_single_code
        register_num = 2
        request_frame = struct.pack('>BBHH',
                                    self.salve,
                                    function_code,
                                    address,
                                    register_num)
        crc = crc16(request_frame)
        request_frame += struct.pack('<H', crc)
        self.ser.write(request_frame)
        response = self.ser.read(9)
        time.sleep(0.01)
        if len(response) == 9:
            _, _, bytes_num, value_low, value_high = struct.unpack('>BBBHH', response[:7])
            if bytes_num == 4:
                value = value_high << 16 | value_low
                return value
        return None

    def close(self):
        self.ser.close()

    def is_open(self):
        return self.ser.is_open


class Motor:
    """
    英鹏飞R485步进电机驱动器控制类
    通过Modbus RTU协议（RS485）控制步进电机

    行程说明：
        下限位 = 0 脉冲（归零点）
        上限位 = UPPER_LIMIT 脉冲
        调用 home() 后坐标系生效
    """

    MOTOR_LIM_MIN =-140000
    MOTOR_LIM_MAX =60000

    UPPER_LIMIT = MOTOR_LIM_MAX    # 上限位脉冲数（实测值）
    LOWER_LIMIT = MOTOR_LIM_MIN         # 下限位脉冲数（归零点）
    TRAVEL      = 240000      # 总行程脉冲数

    def __init__(self, serial_port, serial_baud=115200):
        self.serial = Motor_Serial(serial_port, serial_baud)
        # 临时停用喷头软限位的自动下发。
        # 如需恢复，只需重新启用下面这一行。
        # self._setup_limits()
        # self.set_current_absolute_position(0)
 
    # ==================== 设置记忆性 ====================

    def save_parameters(self):
        """
        发送断电保存指令，将当前参数（包括限位）保存到驱动器
        地址：0x00DC，写入 1
        """
        self.serial.write_single(0x00DC, 1)
        print("参数已保存到驱动器")


    # ==================== 设置限位 ====================

    def set_soft_limit_lower(self, pulse):
        """
        设置软件负限位（绝对位置下限）
        地址：0x006E-0x006F（32位）
        :param pulse: 下限脉冲数（有符号整数），例如 -10000
        """
        self.serial.write_multiple(0x006E, pulse)

    def set_soft_limit_upper(self, pulse):
        """
        设置软件正限位（绝对位置上限）
        地址：0x0070-0x0071（32位）
        :param pulse: 上限脉冲数（有符号整数），例如 245000
        """
        self.serial.write_multiple(0x0070, pulse)


    def _setup_limits(self):
        """设置软件限位（在初始化时自动调用）"""
        try:
            self.set_soft_limit_lower(self.LOWER_LIMIT)
            self.set_soft_limit_upper(self.UPPER_LIMIT)
            # self.save_parameters()
            print(f"✓ 软件限位已设置: 下限={self.LOWER_LIMIT}, 上限={self.UPPER_LIMIT}")
        except Exception as e:
            print(f"限位设置失败: {e}")

    # ==================== 归零与位置 ====================

    def home(self, speed=5):
            """
            归零：运动到下限位并将当前位置设为0
            每次开机后必须先调用此函数建立坐标系

            安全策略：先读取当前位置，计算到下限位的距离，
            避免因 forward(TRAVEL) 超出软限位导致错误。
            """
            print("归零中，向下限位运动...")
            current = self.get_absolute_position()
            if current is None:
                print("警告：无法读取当前位置，尝试运动最大行程")
                distance = self.TRAVEL
            else:
                # 当前位置到下限位的距离（向下运动，脉冲为负）
                distance = current - self.LOWER_LIMIT
                if distance <= 0:
                    print(f"✓ 已在限位附近 (当前={current})，直接设零")
                    self.set_current_absolute_position(0)
                    return
                print(f"  当前位置={current}，需向下运动 {distance} 脉冲到下限位")
            self.set_speed(speed)
            self.forward(distance)
            self.set_current_absolute_position(0)
            print("✓ 归零完成，当前位置 = 0（下限位）")

    def go_to(self, pulse, speed=100):
        """
        移动到指定绝对位置（脉冲数）
        :param pulse: 目标脉冲数，范围 [LOWER_LIMIT, UPPER_LIMIT]
        :param speed: 运动速度 r/min
        """
        if not (self.LOWER_LIMIT <= pulse <= self.UPPER_LIMIT):
            raise ValueError(
                f"目标位置 {pulse} 超出范围 [{self.LOWER_LIMIT}, {self.UPPER_LIMIT}]"
            )
        self.set_speed(speed)
        self.move_to_absolute_position(pulse)

    def go_to_ratio(self, ratio, speed=100):
        """
        按比例移动
        :param ratio: 0.0 = 下限位，1.0 = 上限位
        :param speed: 运动速度 r/min
        """
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"比例 {ratio} 超出范围 [0.0, 1.0]")
        pulse = int(ratio * self.TRAVEL)
        self.go_to(pulse, speed)

    # ==================== 参数读取 ====================

    def get_driver_subdivision(self):
        """获取驱动器细分，默认4000脉冲/转，值域: 200~65535"""
        return self.serial.read_single(0x0007)

    def set_driver_subdivision(self, subdivision):
        """设置驱动器细分"""
        subdivision = max(200, min(65535, subdivision))
        self.serial.write_single(0x0007, subdivision)

    def get_port_timeout(self):
        """读取串口超时时间，单位10ms"""
        return self.serial.read_single(0x0008)

    def set_port_timeout(self, timeout):
        """设置串口超时时间，单位ms"""
        self.serial.write_single(0x0008, int(timeout / 10))

    def get_baud(self):
        """读取当前波特率"""
        baud_map = {
            1: 300, 2: 600, 3: 1200, 4: 2400, 5: 4800,
            6: 9600, 7: 14400, 8: 19200, 9: 38400,
            10: 56000, 11: 57600, 12: 115200,
            13: 230400, 14: 460800, 15: 921600
        }
        value = self.serial.read_single(0x0009)
        return baud_map.get(value, None)

    def set_baud(self, baud):
        """设置波特率"""
        baud_map = {
            300: 1, 600: 2, 1200: 3, 2400: 4, 4800: 5,
            9600: 6, 14400: 7, 19200: 8, 38400: 9,
            56000: 10, 57600: 11, 115200: 12,
            230400: 13, 460800: 14, 921600: 15
        }
        if baud not in baud_map:
            raise ValueError(f"不支持的波特率: {baud}")
        self.serial.write_single(0x0009, baud_map[baud])

    def get_smoothing_constant(self):
        """读取平滑常数，值域: 1~2500"""
        return self.serial.read_single(0x000A)

    def set_smoothing_constant(self, constant):
        """
        设置平滑常数
        数值越小平滑越好但响应越慢，脉冲延时(ms) = 1000/平滑常数
        """
        constant = max(1, min(2500, constant))
        self.serial.write_single(0x000A, constant)

    def get_encoder_resolution(self):
        """读取编码器分辨率"""
        return self.serial.read_single(0x000F)

    def set_encoder_resolution(self, resolution):
        """设置编码器分辨率，值域: 1~65536，单位: 线"""
        self.serial.write_single(0x000F, resolution)

    def get_min_encoder_resolution(self):
        """读取编码器最小分辨率（仅闭环有效）"""
        return self.serial.read_single(0x0017)

    def set_min_encoder_resolution(self, resolution):
        """设置编码器最小分辨率，低于此值时工作于开环模式"""
        self.serial.write_single(0x0017, resolution)

    def get_current_speed(self):
        """读取实际速度（开环为脉冲速度，闭环为转子速度）"""
        return self.serial.read_single(0x0019)

    def get_current_flow(self):
        """读取电机实时电流，返回值单位为A"""
        flow = self.serial.read_single(0x001A)
        return flow / 1000 if flow is not None else None

    def set_input_delay(self, port, delay):
        """
        设置输入接收信号延时时间
        :param port: 端口号 0~7
        :param delay: 延时时间 ms
        """
        self.serial.write_single(0x001B + port, delay)

    def get_direction(self):
        """读取电机运行方向，True=CW（顺时针）"""
        value = self.serial.read_single(0x006B)
        return value == 0

    def set_direction(self, CW=True):
        """设置电机运行方向，CW=True为顺时针"""
        self.serial.write_single(0x006B, 0 if CW else 1)

    def get_target_speed(self):
        """读取目标速度寄存器，单位 r/min"""
        return self.serial.read_single(0x009A)

    def set_speed(self, speed):
        """
        设置速度
        :param speed: 速度，0~10000 r/min
        """
        try:
            speed_value = float(speed)
        except (TypeError, ValueError):
            speed_value = 0.0
        speed_value = max(0.0, min(10000.0, speed_value))

        # The motor speed register is a single WORD, so the transmitted value
        # must remain an integer even when the UI accepts decimal rpm input.
        if speed_value == 0:
            register_speed = 0
        elif speed_value < 1:
            register_speed = 1
        else:
            register_speed = int(speed_value + 0.5)
        self.serial.write_single(0x009A, register_speed)
        return register_speed

    def get_absolute_position(self):
        """
        读取电机当前绝对位置（32位有符号整数）
        地址：0x0004-0x0005
        返回值：有符号整数（脉冲数）
        """
        # 使用已有的 read_multiple 方法（注意其字节顺序已按原设计）
        # 注意：原 read_multiple 返回无符号32位，需要转换为有符号
        val = self.serial.read_multiple(0x0004)
        if val is None:
            return None
        # 转换为有符号（如果最高位为1，则为负数）
        if val >= 0x80000000:
            val = val - 0x100000000
        return val


    # ==================== 运动控制 ====================

    def move(self, CW=True):
        """
        电机持续运行
        :param CW: True=正向，False=反向
        """
        value = 1 if CW else 257
        self.serial.write_single(0x00C8, value)

    def stop(self, emergency=False):
        """
        电机停止
        :param emergency: True=急停，False=减速停止
        """
        value = 256 if emergency else 0
        self.serial.write_single(0x00C8, value)

    def dot_move(self, speed, CW=True):
        """
        电机点动
        :param speed: 0~511 r/min
        :param CW: True=顺时针
        """
        speed = min(511, speed)
        dir_bit = 0 if CW else 1
        value = dir_bit << 15 | speed << 5 | 1
        self.serial.write_single(0x00CA, value)

    def dot_stop(self, emergency=False):
        """电机点动停止"""
        value = emergency << 4
        self.serial.write_single(0x00CA, value)

    def time_move(self, t):
        """
        电机运行一定时间（以set_speed的设置值运动）
        :param t: 时间(秒)，正值正向，负值反向
        """
        time_ms = int(t * 1000)
        self.serial.write_multiple(0x00CC, time_ms)
        time.sleep(abs(t))

    def pulse_move(self, pulse, waiting=True):
        """
        电机运行一定脉冲数（停止状态才能响应）
        :param pulse: 脉冲数，正值正向，负值反向
        :param waiting: True=等待运动完成
        """
        self.serial.write_multiple(0x00CE, pulse)
        if waiting:
            self.reached()

    def pulse_move_immediately(self, pulse, waiting=True):
        """
        电机运行一定脉冲数（任何时候均响应，会中断当前运动）
        :param pulse: 脉冲数，正值正向，负值反向
        :param waiting: True=等待运动完成
        """
        self.serial.write_multiple(0x00DE, pulse)
        if waiting:
            self.reached()

    def move_to_absolute_position(self, pulse, waiting=True):
        """
        运动到指定绝对位置（仅停止时才能执行）
        :param pulse: 目标绝对位置（脉冲数）
        :param waiting: True=等待运动完成
        """
        self.serial.write_multiple(0x00D0, pulse)
        if waiting:
            self.reached()

    def set_current_absolute_position(self, pulse):
        """
        设置电机当前绝对位置（重新设定坐标系，物理位置不变）
        注意：无记忆，断电后丢失，每次开机需重新调用 home()
        :param pulse: 设定值
        """
        self.serial.write_multiple(0x00D2, pulse)

    # ==================== 使能与保存 ====================

    def enable(self):
        """电机使能"""
        self.serial.write_single(0x00D4, 0)

    def disable(self):
        """释放电机"""
        self.serial.write_single(0x00D4, 1)

    def restart(self):
        """驱动器重启"""
        self.serial.write_single(0x00D4, 1 << 8)

    def save(self):
        """保存所有记忆寄存器到驱动器"""
        self.serial.write_single(0x00DC, 1)

    def reset(self):
        """恢复出厂设置"""
        self.serial.write_single(0x00DC, 0)

    # ==================== 状态检测 ====================

    def is_running(self):
        """判断电机是否仍在运动（通过速度寄存器判断）"""
        speed = self.get_current_speed()
        return speed is not None and speed > 0

    def reached(self):
        """等待电机运动完成"""
        time.sleep(0.2)
        while True:
            time.sleep(0.1)
            if not self.is_running():
                return

    # ==================== 快捷方向运动 ====================

    def forward(self, pulse, speed=100, waiting=True):
        """向正方向运动指定脉冲数,即向下"""
        self.set_speed(speed)
        self.pulse_move_immediately(-pulse, waiting)

    def backward(self, pulse, speed=100, waiting=True):
        """向负方向运动指定脉冲数,即向上,为正脉冲"""
        self.set_speed(speed)
        self.pulse_move_immediately(pulse, waiting)

    # ==================== 串口管理 ====================

    def close(self):
        """关闭串口"""
        self.serial.close()

    def is_open(self):
        """串口是否已打开"""
        return self.serial.is_open()
    

if __name__ == '__main__':

    motor = Motor('COM5', 115200)
    motor.enable()
    # motor.go_to(40000)
    motor.forward(2000)
    print(motor.get_absolute_position())
    # motor.set_speed(1)
    # motor.move(CW=False)#用了move后一定要加motor.stop(),stop用于停止运动，而close是用来关串口的，即使串口关了也不会停止
    # time.sleep(3)
    # motor.stop()
    motor.close()
#注意！！：forward是向下运动，CW=True是向上运动，越下方脉冲数越小，即为负数
