import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QButtonGroup
from PyQt5.QtCore import QTimer
from motor_ui import Ui_MainWindow
from serial.tools import list_ports
from motor_serial import Motor, Motor_Serial

bauds = [300, 600, 1200, 2400, 4800, 9600, 11400, 19200,
         38400, 56000, 57600, 115200, 230400, 460800, 921600]
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.motor = None
        self.motor_state = False
        self.CW = True

        available_ports = list(list_ports.comports())
        ports = sorted(available_ports, key=lambda x: int(x.device[3:]))
        for port in ports:
            self.comboBox_port.addItem(port.device)
            if port.description.split(' (')[0] == 'USB Serial Port':
                self.serial_port = port.device
                self.comboBox_port.setCurrentIndex(ports.index(port))
        self.comboBox_port.activated.connect(self.choose_port)

        for baud in bauds:
            self.comboBox_baud.addItem(str(baud))
        self.comboBox_baud.setCurrentIndex(11)
        self.serial_baud = bauds[11]
        self.comboBox_baud.activated.connect(self.choose_baud)

        self.lamp_serial.setStyleSheet('background-color: red; border-radius: 8px')
        self.pb_open_serial.clicked.connect(self.open_serial)

        self.lamp_motor_state.setStyleSheet('background-color: gray; border-radius: 8px')

        self.radioButton_cw.setChecked(True)
        self.buttonGroup = QButtonGroup()
        self.buttonGroup.addButton(self.radioButton_cw)
        self.buttonGroup.addButton(self.radioButton_ccw)
        self.radioButton_cw.clicked.connect(self.change_direction)
        self.radioButton_ccw.clicked.connect(self.change_direction)

        self.pb_start_motor.clicked.connect(self.start_motor)
        self.pb_stop_motor.clicked.connect(self.stop_motor)
        self.pb_set_zero.clicked.connect(self.set_zero)
        self.pb_move_to_zero.clicked.connect(self.move_to_zero)
        self.pb_reset_motor.clicked.connect(self.restart_motor)
        self.pb_set_speed.clicked.connect(self.set_speed)

        self.pb_dot_move.clicked.connect(self.dot_move)
        self.pb_pulse_move.clicked.connect(self.pulse_move)
        self.pb_move_to_position.clicked.connect(self.move_to_position)
        self.pb_time_move.clicked.connect(self.time_move)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout_callback)

        self.serial = None
        self.target_speed = None
        self.current_speed = None

    def choose_port(self):
        self.serial_port = self.comboBox_port.currentText()

    def choose_baud(self):
        index = self.comboBox_baud.currentIndex()
        self.serial_baud = bauds[index]

    def open_serial(self):
        if self.pb_open_serial.text() == '打开串口':
            # self.serial = Motor_Serial(self.serial_port, self.serial_baud)
            self.lamp_serial.setStyleSheet('background-color: rgb(102, 255, 0); border-radius: 8px')
            self.pb_open_serial.setText('关闭串口')

            self.motor = Motor(self.serial_port, self.serial_baud)
            self.serial = self.motor.serial
            self.timer.start(200)
        else:
            self.serial.close()
            self.lamp_serial.setStyleSheet('background-color: red; border-radius: 8px')
            self.lamp_motor_state.setStyleSheet('background-color: gray; border-radius: 8px')
            self.label_motor_state.setText('断开')
            self.pb_open_serial.setText('打开串口')
            self.timer.stop()

    def timeout_callback(self):
        current_position = self.motor.get_absolute_position()
        self.target_speed = self.motor.get_target_speed()
        self.current_speed = self.motor.get_current_speed()
        voltage = self.motor.get_bus_voltage()
        flow = self.motor.get_current_flow()
        _, _, _, self.motor_state = self.motor.get_work_state()
        if self.motor_state == 0x11:
            self.lamp_motor_state.setStyleSheet('background-color: rgb(102, 255, 0); border-radius: 8px')
            self.label_motor_state.setText('运行')
        else:
            self.lamp_motor_state.setStyleSheet('background-color: red; border-radius: 8px')
            self.label_motor_state.setText('空闲')
        self.lineEdit_current_position.setText(str(current_position))
        self.lineEdit_target_speed.setText(str(self.target_speed))
        self.lineEdit_current_speed.setText(str(self.current_speed))
        self.lineEdit_bus_voltage.setText(str(voltage))
        self.lineEdit_current_flow.setText(str(flow))

    def change_direction(self):
        if self.radioButton_ccw.isChecked():
            self.CW = False
        elif self.radioButton_cw.isChecked():
            self.CW = True

    def start_motor(self):
        if self.motor.serial.is_open():
            try:
                self.motor.move(self.CW)
            except Exception as e:
                print(e)

    def stop_motor(self):
        if self.motor.serial.is_open():
            try:
                self.motor.stop()
            except Exception as e:
                print(e)

    def set_zero(self):
        if self.motor.serial.is_open():
            try:
                self.motor.set_current_absolute_position(0)
            except Exception as e:
                print(e)

    def move_to_zero(self):
        if self.motor.serial.is_open():
            try:
                self.motor.move_to_absolute_position(0)
            except Exception as e:
                print(e)

    def restart_motor(self):
        if self.motor.serial.is_open():
            try:
                self.motor.restart()
            except Exception as e:
                print(e)

    def set_speed(self):
        if self.motor.serial.is_open():
            text = self.lineEdit_set_speed.text()
            try:
                number = int(text)
                self.motor.set_speed(number)
            except ValueError as e:
                print(e)

    def dot_move(self):
        if self.motor.serial.is_open():
            text = self.lineEdit_dot_move.text()
            try:
                speed = int(text)
                self.motor.dot_move(speed, self.CW)
            except ValueError as e:
                print(e)

    def pulse_move(self):
        if self.motor.serial.is_open():
            text = self.lineEdit_pulse_move.text()
            try:
                pulse = int(text)
                if not self.CW:
                    pulse = -pulse
                self.motor.pulse_move(pulse)
            except ValueError as e:
                print(e)

    def move_to_position(self):
        if self.motor.serial.is_open():
            text = self.lineEdit_move_to_position.text()
            try:
                position = int(text)
                self.motor.move_to_absolute_position(position)
            except ValueError as e:
                print(e)

    def time_move(self):
        if self.motor.serial.is_open():
            text = self.lineEdit_time_move.text()
            try:
                running_time = float(text)
                if not self.CW:
                    running_time = -running_time
                self.motor.time_move(running_time)
            except ValueError as e:
                print(e)


if __name__ == '__main__':
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
