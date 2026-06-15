import serial
import struct
import time


def crc16(data):
    # 计算CRC校验码
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


class Motor_Serial:
    def __init__(self, serial_port, serial_baud=115200, timeout=2):
        self.ser = serial.Serial(serial_port, serial_baud, timeout=timeout)
        self.salve = 1
        self.read_single_code = 0x03
        self.read_multiple_code = 0x04
        self.write_single_code = 0x06
        self.write_multiple_code = 0x10

    def write_single(self, address, value):
        # 8个字节
        # 7: 站号，01
        # 6: 功能码：03
        # 5-4: 起始地址高位/低位
        # 3-2: 总寄存器数高位/低位
        # 1-0: CRC校验低位/高位
        function_code = self.write_single_code
        if value < 0:
            value = (1 << 16) + value
        # > 表示大端序(Big-Endian)
        # B 表示一个字节，用于表示站号和功能码
        # H 表示两个字节，用于表示寄存器地址和设置的值
        request_frame = struct.pack('>BBHH',
                                    self.salve,
                                    function_code,
                                    address,
                                    value)
        crc = crc16(request_frame)
        # CRC校验码是低字节序格式，因此用 < 表示小端序(Little-Endian)
        request_frame += struct.pack('<H', crc)
        self.ser.write(request_frame)
        self.ser.read(8)

    def read_single(self, address):
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
        # 解析响应帧
        if len(response) == 7:
            _, _, num_bytes, value = struct.unpack('>BBBH', response[:5])
            if num_bytes == 2:
                return value
        return None

    def write_multiple(self, address, value):
        # 13个字节
        # 12: 站号，01
        # 11: 功能码：10
        # 10-9: 起始地址高位/低位
        # 8-7: 寄存器总数：00 02
        # 6: 总字节数：04
        # 5-4: 寄存器值高位/低位
        # 3-2: 寄存器值高位/低位
        # 1-0: CRC校验低位/高位
        function_code = self.write_multiple_code
        register_num = 2
        bytes_num = 4
        if value < 0:
            value = (1 << 32) + value
        value_high = value >> 16
        value_low = value & 0xffff
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
        response = self.ser.read(8)
        time.sleep(0.01)

    def read_multiple(self, address):
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

        # 解析响应帧
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
    def __init__(self, serial_port, serial_baud=115200):
        self.serial = Motor_Serial(serial_port, serial_baud)

    def get_absolute_position(self):
        """
        读取电机当前实际位置
        地址：0x0004~0x0005, 写入非法
        :return:
        """
        address = 0x0004
        return self.serial.read_multiple(address)

    def get_work_state(self):
        """
        读取运行及输入口状态
        地址: 0x0006, 写入非法
        # bit15: 保留，恒为0
        # 14-13: 软件正/负限位标识，为 1 时表示到达软件正/负限位
        # 12: 到位输出标识，运行时为 0，到位时为 1
        # 11: 位置提醒标识，可设定大于或小于，超过位置时为 1
        # 10: 位置超差警告，转子位置和命令位置超过0x0010设定值时为 1
        # 9-8: 运行状态，00 表示电机空闲，01 表示电机即将启动，10 表示电机即将停止，11 表示电机正在运行
        # 7-0: X7-X0输入状态，1 为高电平，0 为低电平
        :return:
        """
        address = 0x0006
        value = self.serial.read_single(address)
        positive_limit = value >> 14                # 正限位
        negative_limit = (value >> 13) & 1          # 负限位
        on_stop = (value >> 12) & 1                 # 停止状态
        offside = (value >> 11) & 1                 # 越位标志
        off_tolerance = (value >> 10) & 1           # 超差标志
        running_state = (value >> 8) & 0x11         # 运行状态，00为空闲，01为即将启动，10为即将停止，11为正在运行
        input7_state = (value >> 7) & 1             # X7输入状态
        input6_state = (value >> 6) & 1
        input5_state = (value >> 5) & 1
        input4_state = (value >> 4) & 1
        input3_state = (value >> 3) & 1
        input2_state = (value >> 2) & 1
        input1_state = (value >> 1) & 1
        input0_state = value & 1

        return [positive_limit, negative_limit], on_stop, [offside, off_tolerance], running_state

    def get_diver_subdivision(self):
        "读取驱动器当前细分设置，细分越大，电机转动越平稳，值域：200-65535"
        address = 0x0007
        return self.serial.read_single(address)

    def set_diver_subdivision(self, division):
        """
        设置驱动器细分
        地址: 0x0007, 记忆
        值域: 200~65535, 脉冲/r
        :param division:
        :return:
        """
        address = 0x0007
        if division < 200:
            division = 200
        elif division > 65535:
            division = 65535
        self.serial.write_single(address, division)

    def get_timeout(self):
        address = 0x0008
        return self.serial.read_single(address)

    def set_timeout(self, timeout):
        """
        设置串口超时时间
        地址: 0x0008, 记忆
        值域: 0~65535, 单位 10ms
        :param timeout: 单位 ms
        :return:
        """
        address = 0x0008
        timeout = int(timeout / 10)
        self.serial.write_single(address, timeout)

        def get_baud(self):
            address = 0x0009
            value = self.serial.read_single(address)
            if value == 1:
                baud = 300
            elif value == 2:
                baud = 600
            elif value == 3:
                baud = 1200
            elif value == 4:
                baud = 2400
            elif value == 5:
                baud = 4800
            elif value == 6:
                baud = 9600
            elif value == 7:
                baud = 14400
            elif value == 8:
                baud = 19200
            elif value == 9:
                baud = 38400
            elif value == 10:
                baud = 56000
            elif value == 11:
                baud = 57600
            elif value == 12:
                baud = 115200
            elif value == 13:
                baud = 230400
            elif value == 14:
                baud = 460800
            elif value == 15:
                baud = 921600
            return baud

    def set_baud(self, baud):
        """
        设置驱动器波特率
        地址: 0x0009, 记忆
        值域: 1~15, 1:300, 2:600, 3:1200, 4:2400, 5:4800, 6:9600, 7:14400, 8:19200, 9:38400
                   10:56000, 11:57600, 12:115200, 13:230400, 14:460800, 15:921600
        :param baud:
        :return:
        """
        address = 0x0009
        if baud == 300:
            value = 1
        elif baud == 600:
            value = 2
        elif baud == 1200:
            value = 3
        elif baud == 2400:
            value = 4
        elif baud == 4800:
            value = 5
        elif baud == 9600:
            value = 6
        elif baud == 14400:
            value = 7
        elif baud == 19200:
            value = 8
        elif baud == 38400:
            value = 9
        elif baud == 56000:
            value = 10
        elif baud == 57600:
            value = 11
        elif baud == 115200:
            value = 12
        elif baud == 230400:
            value = 13
        elif baud == 460800:
            value = 14
        elif baud == 921600:
            value = 15
        else:
            raise ValueError('Error baud rate, baud rate can only be selected from:'
                             '300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 38400, 56000, '
                             '57600, 115200, 23400, 460800 and 921600')
        self.serial.write_single(address, value)

    def get_smooth_constant(self)   :
        address = 0x000a
        return self.serial.read_single(address)

    def set_smooth_constant(self, constant):
        """
        设置平滑常数
        地址: 0x000A, 记忆
        值域: 1~2500
        数值越小，平滑越好，脉冲延时越长，相应越慢；数值越大，平滑越差，脉冲延时越小，响应越快
        脉冲延时(ms) = 1000/平滑常数
        :param constant:
        :return:
        """
        address = 0x000a
        if constant < 1:
            constant = 1
        elif constant > 2500:
            constant = 2500
        self.serial.write_single(address, constant)

    def get_encoder_resolution(self):
        address = 0x000f
        return self.serial.read_single(address)

    def set_encoder_resolution(self, resolution):
        """
        设置编码器分辨率
        地址: 0x000F, 记忆
        值域: 1~65536, 单位: 线
        :param resolution:
        :return:
        """
        address = 0x000f
        self.serial.write_single(address, resolution)

    def get_min_encoder_resolution(self):
        address = 0x0017
        return self.serial.read_single(address)

    def set_min_encoder_resolution(self, resolution):
        """
        设置编码器最小分辨率(仅闭环有效), 如果实际的编码器数小于这个数，工作于开环模式
        地址: 0x0017
        值域: 1~65536
        :param resolution:
        :return:
        """
        address = 0x0017
        self.serial.write_single(address, resolution)

    def get_current_speed(self):
        """
        读取实际速度(开环时为脉冲速度，闭环时为转子速度)
        地址: 0x0019, 写入非法
        :return:
        """
        address = 0x0019
        return self.serial.read_single(address)

    def get_current_flow(self):
        """
        读取电机实时电流, 单位 mA
        地址: 0x001A, 写入非法
        :return:
        """
        address = 0x001a
        flow = self.serial.read_single(address)
        return flow / 1000

    def get_bus_voltage(self):
        """
        读取总线电压的最大值
        最大值随时间变化，只显示测得的最大值，如果需要重新测试，需要写0清楚当前最大值
        地址: 0x0044
        :return:
        """
        address = 0x0044
        voltage = self.serial.read_single(address)
        return voltage / 100

    def set_input_delay(self, port, delay):
        """
        设置输入接收信号延时时间
        地址: 0x001B~0x0022, 记忆
        值域: 0~65536, ms
        :param port: 0~7
        :param delay:
        :return:
        """
        address = 0x001b + port
        self.serial.write_single(address, delay)

    def get_direction(self):
        address = 0x006b
        value = self.serial.read_single(address)
        if value == 0:
            return True         # 正向
        else:
            return False

    def set_direction(self, CW=True):
        """
        设置电机运行方向
        地址: 0x006B, 记忆
        值域: 0~1, 0表示 CW，1表示 CCW
        :param CW: True or False
        :return:
        """
        address = 0x006b
        if CW:
            value = 0
        else:
            value = 1
        self.serial.write_single(address, value)

    def turn_over_input(self, port):
        """
        翻转输入口电平信号
        地址: 0x006C, 无记忆，读取非法
        :param port: 0~7
        :return:
        """
        address = 0x006c
        value = 1 << port
        self.serial.write_single(address, value)

    def get_target_speed(self):
        address = 0x009a
        return self.serial.read_single(address)

    def set_speed(self, speed):
        """
        设置速度
        地址：0x009A
        :param speed: 速度，0~10000r/min
        :return:
        """
        address = 0x009A
        if speed > 10000:
            speed = 10000
        self.serial.write_single(address, speed)

    def move(self, CW=True):
        """
        电机启动
        地址：0x00C8, 无记忆
        值域: 0, 1, 256, 257, 0减速停止, 1正向运行，256急停，257反向运行
        :param CW: True or False
        :return:
        """
        address = 0x00C8
        if CW:
            value = 1
        else:
            value = 257
        self.serial.write_single(address, value)

    def stop(self, emergency=False):
        """
        电机停止
        地址：0x00C8, 无记忆
        值域: 0, 1, 256, 257, 0减速停止, 1正向运行，256急停，257反向运行
        :param emergency: 1急停，0减速停止
        :return:
        """
        address = 0x00C8
        if emergency:
            value = 256
        else:
            value = 0
        self.serial.write_single(address, value)

    def dot_move(self, speed, CW=True):
        """
        电机点动
        地址: 0x00CA, 无记忆
        bit 15: 点动方向，0为 CW, 1为 CCW
        14-6: 点动速度
        5: 点动停止方式，0为减速停止，1为立即停止，启动时无意义
        4-1: 保留
        0: 点动启动与停止，0为停止，1为启动
        :param speed: 0~511 r/min
        :param CW: 顺时针
        :return:
        """
        address = 0x00ca
        if CW:
            dir = 0
        else:
            dir = 1
        if speed > 511:
            speed = 511
        value = dir << 15 | speed << 5 | 1
        self.serial.write_single(address, value)

    def dot_stop(self, emergency=False):
        """
        电机点动停止
        """
        address = 0x00ca
        value = emergency << 4
        self.serial.write_single(address, value)

    def time_move(self, t):
        """
        电机运行一定的时间
        地址: 0x00CC-0x00CD
        :param time: s, 值为正则方向正，值为负则方向负
        :return:
        """
        address = 0x00cc
        time_ms = int(t * 1000)
        self.serial.write_multiple(address, time_ms)
        if t < 0:
            t = -t
        time.sleep(t)

    def pulse_move(self, pulse, waiting=True):
        """
        电机运行一定的脉冲数, 停止状态才能响应
        地址: 0x00CE-0x00CF
        :param pulse: 脉冲数，值为正时方向正，值为负时方向负
        :return:
        """
        address = 0x00ce
        self.serial.write_multiple(address, pulse)

        if waiting:
            self.reached()

    def pulse_move_immediately(self, pulse, waiting=True):
        """
        电机运行一定的脉冲数，任何时候都会响应，响应时其他运行指令会强制结束
        地址: 0x00DE-0x00DF
        :param pulse: 脉冲数
        :return:
        """
        address = 0x00de
        self.serial.write_multiple(address, pulse)

        if waiting:
            self.reached()

    def move_to_absolute_position(self, pulse, waiting=True):
        """
        运动到指定的绝对位置，仅停止时才能执行
        地址: 0x00D0-0x00D1
        :param pulse: 指定的绝对位置，单位脉冲数，小于当前位置方向为负，大于当前位置方向为正
        :return:
        """
        address = 0x00d0
        self.serial.write_multiple(address, pulse)

        if waiting:
            self.reached()

    def move_to_absolute_position_immediately(self, pulse, waiting=True):
        """
        运动到指定绝对位置，接收到该指令立即执行，当前指令强制结束
        地址: 0x00E8-0x00E9
        :param pulse:
        :return:
        """
        address = 0x00e8
        self.serial.write_multiple(address, pulse)

        if waiting:
            self.reached()

    def set_current_absolute_position(self, pulse):
        """
        设置电机当前绝对位置
        地址: 0x00D2-0x00D3, 无记忆
        :param pulse: 设定当前绝对位置，物理位置不变，实际位置寄存器值改变
        :return:
        """
        address = 0x00d2
        self.serial.write_multiple(address, pulse)

    def enable(self):
        """
        电机使能
        地址: 0x00D4
        bit 15-8: 0~1，写 1 驱动重启
             7-0: 0~1, 写 0 马达使能，写 1 释放马达
        :return:
        """
        address = 0x00d4
        value = 0
        self.serial.write_single(address, value)

    def disable(self):
        # 释放马达
        address = 0x00d4
        value = 1
        self.serial.write_single(address, value)

    def restart(self):
        # 驱动重启
        address = 0x00d4
        value = 1 << 8
        self.serial.write_single(address, value)

    def save(self):
        """
        保存所有记忆寄存器
        地址: 0x00DC
        值域: 0~1, 1时保存，0时恢复出厂设置
        :return:
        """
        address = 0x00dc
        value = 1
        self.serial.write_single(address, value)

    def reset(self):
        """
        恢复出厂设置
        地址: 0x00DC
        :return:
        """
        address = 0x00dc
        value = 0
        self.serial.write_single(address, value)

    def reached(self):
        while True:
            time.sleep(0.1)
            if not self.is_running():
                return
  
    def is_running(self):
        _, on_stop, _, _ = self.get_work_state()
        return not on_stop

    def forward(self, pulse, speed=100, waiting=True):
        self.set_speed(speed)
        self.pulse_move_immediately(-pulse, waiting)

    def backward(self, pulse, speed=100, waiting=True):
        self.set_speed(speed)
        self.pulse_move_immediately(pulse, waiting)




if __name__ == '__main__':
    motor = Motor('COM6', 115200)
    motor.enable
    # motor.set_speed(5)
    # print(motor.get_target_speed())
    # time.sleep(1)
    # motor.move()
    # time.sleep(30)
    motor.backward(20000)

    # motor.set_speed(10)
    # motor.move(False)
    motor.stop()
    

