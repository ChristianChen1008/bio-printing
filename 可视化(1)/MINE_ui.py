# -*- coding: utf-8 -*-
"""
3D打印综合控制台 - UI 界面定义
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PrintWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("PrintWindow")
        MainWindow.resize(1220, 856)
        MainWindow.setMinimumSize(QtCore.QSize(1160, 796))
        MainWindow.setWindowTitle("3D打印综合控制台")

        font_main = QtGui.QFont("Microsoft YaHei", 9)
        font_mono = QtGui.QFont("Consolas", 9)
        font_title = QtGui.QFont("Microsoft YaHei", 15, QtGui.QFont.Bold)
        font_group = QtGui.QFont("Microsoft YaHei", 9, QtGui.QFont.Bold)
        font_button = QtGui.QFont("Microsoft YaHei", 10, QtGui.QFont.Bold)
        font_state = QtGui.QFont("Microsoft YaHei", 13, QtGui.QFont.Bold)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setFont(font_main)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("""
            QWidget#centralwidget {
                background: #f4f7fb;
                color: #172033;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9e2ef;
                border-radius: 8px;
                margin-top: 18px;
                padding: 12px 10px 10px 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #20324d;
            }
            QLabel {
                color: #34445f;
            }
            QLineEdit, QComboBox {
                min-height: 24px;
                padding: 2px 7px;
                border: 1px solid #c8d3e3;
                border-radius: 5px;
                background: #fbfdff;
                selection-background-color: #4f7cff;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #4f7cff;
                background: #ffffff;
            }
            QLineEdit:disabled {
                background: #edf1f6;
                color: #8b97aa;
            }
            QPushButton {
                min-height: 26px;
                padding: 4px 10px;
                border: 1px solid #c7d2e2;
                border-radius: 6px;
                background: #ffffff;
                color: #24344d;
            }
            QPushButton:hover {
                background: #edf3ff;
                border-color: #8fb0ff;
            }
            QPushButton:pressed {
                background: #dce8ff;
            }
            QTextEdit {
                border: 1px solid #d9e2ef;
                border-radius: 7px;
                background: #0f1724;
                color: #d7e2f1;
                padding: 8px;
            }
            QCheckBox {
                spacing: 7px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
        """)

        self.header = QtWidgets.QFrame(self.centralwidget)
        self.header.setGeometry(QtCore.QRect(14, 10, 1192, 58))
        self.header.setObjectName("header")
        self.header.setStyleSheet("""
            QFrame#header {
                background: #182235;
                border-radius: 10px;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        self.label_title = QtWidgets.QLabel("3D打印综合控制台", self.header)
        self.label_title.setGeometry(18, 10, 260, 26)
        self.label_title.setFont(font_title)
        self.label_subtitle = QtWidgets.QLabel("UR3机械臂 / 步进电机喷头 / 轨迹预览与打印控制", self.header)
        self.label_subtitle.setGeometry(20, 35, 520, 18)
        self.label_subtitle.setStyleSheet("color:#b9c6dc;")
        self.label_print_state = QtWidgets.QLabel("就绪", self.header)
        self.label_print_state.setGeometry(1070, 12, 92, 34)
        self.label_print_state.setAlignment(QtCore.Qt.AlignCenter)
        self.label_print_state.setFont(font_state)
        self.label_print_state.setStyleSheet(
            "background:#e7f7ef;color:#147a45;border-radius:17px;padding:4px 12px;"
        )

        # ==================== 左侧控制区 ====================
        self.group_arm = QtWidgets.QGroupBox(self.centralwidget)
        self.group_arm.setGeometry(QtCore.QRect(14, 80, 500, 148))
        self.group_arm.setFont(font_group)
        self.group_arm.setTitle("机械臂设置")

        self.label_ip = QtWidgets.QLabel("IP地址", self.group_arm)
        self.label_ip.setGeometry(16, 30, 70, 22)
        self.edit_ip = QtWidgets.QLineEdit("169.254.45.1", self.group_arm)
        self.edit_ip.setGeometry(92, 28, 150, 28)

        self.label_speed = QtWidgets.QLabel("速度", self.group_arm)
        self.label_speed.setGeometry(270, 30, 42, 22)
        self.edit_speed = QtWidgets.QLineEdit("10", self.group_arm)
        self.edit_speed.setGeometry(312, 28, 62, 28)
        self.label_speed_unit = QtWidgets.QLabel("mm/s", self.group_arm)
        self.label_speed_unit.setGeometry(382, 31, 44, 20)

        self.label_accel = QtWidgets.QLabel("加速度", self.group_arm)
        self.label_accel.setGeometry(16, 68, 70, 22)
        self.edit_accel = QtWidgets.QLineEdit("1000", self.group_arm)
        self.edit_accel.setGeometry(92, 68, 86, 28)
        self.label_accel_unit = QtWidgets.QLabel("mm/s²", self.group_arm)
        self.label_accel_unit.setGeometry(186, 71, 54, 20)

        self.label_blend = QtWidgets.QLabel("交融半径", self.group_arm)
        self.label_blend.setGeometry(270, 70, 70, 22)
        self.edit_blend = QtWidgets.QLineEdit("0", self.group_arm)
        self.edit_blend.setGeometry(340, 68, 70, 28)
        self.label_blend_unit = QtWidgets.QLabel("m", self.group_arm)
        self.label_blend_unit.setGeometry(418, 71, 20, 20)

        self.pb_arm_connect = QtWidgets.QPushButton("连接机械臂", self.group_arm)
        self.pb_arm_connect.setGeometry(16, 110, 122, 30)
        self.pb_lift = QtWidgets.QPushButton("抬高到安全位", self.group_arm)
        self.pb_lift.setGeometry(154, 110, 122, 30)
        self.pb_lift.setToolTip("移动机械臂到安全高度 (100, 100, 50)")

        self.lamp_arm = QtWidgets.QLabel(self.group_arm)
        self.lamp_arm.setGeometry(318, 118, 14, 14)
        self.lamp_arm.setStyleSheet("background-color:#9aa5b1;border-radius:7px")
        self.label_arm_state = QtWidgets.QLabel("未连接", self.group_arm)
        self.label_arm_state.setGeometry(338, 113, 86, 24)

        self.label_tcp = QtWidgets.QLabel("TCP: X=-- Y=-- Z=--", self.group_arm)
        self.label_tcp.setGeometry(16, 8, 462, 18)
        self.label_tcp.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_tcp.setFont(font_mono)
        self.label_tcp.setStyleSheet("color:#607089;")

        self.group_ext = QtWidgets.QGroupBox(self.centralwidget)
        self.group_ext.setGeometry(QtCore.QRect(14, 240, 500, 214))
        self.group_ext.setFont(font_group)
        self.group_ext.setTitle("喷头设置")

        self.label_port = QtWidgets.QLabel("串口", self.group_ext)
        self.label_port.setGeometry(16, 30, 44, 22)
        self.combo_port = QtWidgets.QComboBox(self.group_ext)
        self.combo_port.setGeometry(64, 28, 96, 28)

        self.label_baud = QtWidgets.QLabel("波特率", self.group_ext)
        self.label_baud.setGeometry(180, 30, 54, 22)
        self.combo_baud = QtWidgets.QComboBox(self.group_ext)
        self.combo_baud.setGeometry(236, 28, 106, 28)
        self.combo_baud.addItems(["9600", "14400", "19200", "38400", "56000", "57600", "115200"])
        self.combo_baud.setCurrentText("115200")

        self.pb_ext_connect = QtWidgets.QPushButton("连接喷头", self.group_ext)
        self.pb_ext_connect.setGeometry(366, 27, 106, 30)
        self.lamp_ext = QtWidgets.QLabel(self.group_ext)
        self.lamp_ext.setGeometry(366, 74, 14, 14)
        self.lamp_ext.setStyleSheet("background-color:#9aa5b1;border-radius:7px")
        self.label_ext_state = QtWidgets.QLabel("未连接", self.group_ext)
        self.label_ext_state.setGeometry(386, 69, 82, 24)

        self.label_ext_rpm = QtWidgets.QLabel("挤出速度", self.group_ext)
        self.label_ext_rpm.setGeometry(16, 72, 70, 22)
        self.edit_ext_rpm = QtWidgets.QLineEdit("1", self.group_ext)
        self.edit_ext_rpm.setGeometry(92, 72, 64, 28)
        self.label_ext_rpm_unit = QtWidgets.QLabel("r/min", self.group_ext)
        self.label_ext_rpm_unit.setGeometry(164, 75, 48, 20)

        self.label_delay = QtWidgets.QLabel("启动延迟", self.group_ext)
        self.label_delay.setGeometry(236, 74, 70, 22)
        self.edit_delay = QtWidgets.QLineEdit("1", self.group_ext)
        self.edit_delay.setGeometry(306, 72, 48, 28)
        self.label_delay_unit = QtWidgets.QLabel("s", self.group_ext)
        self.label_delay_unit.setGeometry(362, 75, 18, 20)

        self.label_prime = QtWidgets.QLabel("预挤出", self.group_ext)
        self.label_prime.setGeometry(16, 116, 70, 22)
        self.edit_prime = QtWidgets.QLineEdit("2000", self.group_ext)
        self.edit_prime.setGeometry(92, 114, 82, 28)
        self.label_prime_unit = QtWidgets.QLabel("脉冲", self.group_ext)
        self.label_prime_unit.setGeometry(182, 117, 42, 20)

        self.label_retract = QtWidgets.QLabel("回抽", self.group_ext)
        self.label_retract.setGeometry(236, 116, 45, 22)
        self.edit_retract = QtWidgets.QLineEdit("2000", self.group_ext)
        self.edit_retract.setGeometry(282, 114, 82, 28)
        self.label_retract_unit = QtWidgets.QLabel("脉冲", self.group_ext)
        self.label_retract_unit.setGeometry(372, 117, 42, 20)

        self.label_prestop = QtWidgets.QLabel("预停距离", self.group_ext)
        self.label_prestop.setGeometry(16, 160, 70, 22)
        self.edit_prestop = QtWidgets.QLineEdit("0", self.group_ext)
        self.edit_prestop.setGeometry(92, 158, 64, 28)
        self.edit_prestop.setToolTip("路径剩余多少mm时提前停止挤出，0=不启用")
        self.label_prestop_unit = QtWidgets.QLabel("mm", self.group_ext)
        self.label_prestop_unit.setGeometry(164, 161, 30, 20)

        self.pb_ext_start = QtWidgets.QPushButton("启动挤出", self.group_ext)
        self.pb_ext_start.setGeometry(236, 157, 86, 30)
        self.pb_ext_stop = QtWidgets.QPushButton("停止", self.group_ext)
        self.pb_ext_stop.setGeometry(332, 157, 66, 30)
        self.pb_ext_retract = QtWidgets.QPushButton("回抽", self.group_ext)
        self.pb_ext_retract.setGeometry(408, 157, 64, 30)

        self.label_ext_pos = QtWidgets.QLabel("位置: --  速度: -- r/min", self.group_ext)
        self.label_ext_pos.setGeometry(16, 8, 462, 18)
        self.label_ext_pos.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_ext_pos.setFont(font_mono)
        self.label_ext_pos.setStyleSheet("color:#607089;")

        self.group_path = QtWidgets.QGroupBox(self.centralwidget)
        self.group_path.setGeometry(QtCore.QRect(14, 466, 500, 288))
        self.group_path.setFont(font_group)
        self.group_path.setTitle("轨迹参数")

        self.label_path_mode = QtWidgets.QLabel("模式", self.group_path)
        self.label_path_mode.setGeometry(16, 30, 42, 22)
        self.combo_path_mode = QtWidgets.QComboBox(self.group_path)
        self.combo_path_mode.setGeometry(58, 28, 126, 28)
        self.combo_path_mode.addItems(["单条轨迹", "复合程序"])

        self.label_program_preset = QtWidgets.QLabel("预设程序", self.group_path)
        self.label_program_preset.setGeometry(202, 30, 60, 22)
        self.combo_program_preset = QtWidgets.QComboBox(self.group_path)
        self.combo_program_preset.setGeometry(266, 28, 206, 28)
        self.combo_program_preset.addItems(["矩形 + 圆形"])
        self.combo_program_preset.setEnabled(False)

        self.label_type = QtWidgets.QLabel("轨迹类型", self.group_path)
        self.label_type.setGeometry(16, 66, 70, 22)
        self.combo_type = QtWidgets.QComboBox(self.group_path)
        self.combo_type.setGeometry(86, 64, 112, 28)
        self.combo_type.addItems(["矩形填充", "矩形轮廓", "圆形", "弦图", "直线"])
        self.combo_type.setCurrentIndex(1)

        self.label_center = QtWidgets.QLabel("中心点", self.group_path)
        self.label_center.setGeometry(216, 66, 54, 22)
        self.edit_cx = QtWidgets.QLineEdit("105", self.group_path)
        self.edit_cx.setGeometry(276, 64, 60, 28)
        self.edit_cy = QtWidgets.QLineEdit("105", self.group_path)
        self.edit_cy.setGeometry(344, 64, 60, 28)

        self.group_rect = QtWidgets.QGroupBox("矩形", self.group_path)
        self.group_rect.setGeometry(16, 102, 464, 70)
        self.label_w = QtWidgets.QLabel("宽", self.group_rect)
        self.label_w.setGeometry(12, 28, 26, 20)
        self.edit_width = QtWidgets.QLineEdit("30", self.group_rect)
        self.edit_width.setGeometry(42, 28, 62, 28)
        self.label_h = QtWidgets.QLabel("高", self.group_rect)
        self.label_h.setGeometry(122, 31, 26, 20)
        self.edit_height = QtWidgets.QLineEdit("30", self.group_rect)
        self.edit_height.setGeometry(148, 28, 62, 28)
        self.label_lw = QtWidgets.QLabel("线宽", self.group_rect)
        self.label_lw.setGeometry(230, 31, 42, 20)
        self.edit_linewidth = QtWidgets.QLineEdit("1.0", self.group_rect)
        self.edit_linewidth.setGeometry(272, 28, 62, 28)
        self.label_corner = QtWidgets.QLabel("起始角", self.group_rect)
        self.label_corner.setGeometry(352, 31, 54, 20)
        self.combo_corner = QtWidgets.QComboBox(self.group_rect)
        self.combo_corner.setGeometry(406, 28, 48, 28)
        self.combo_corner.addItems(["bl", "br", "tl", "tr"])

        self.group_circle = QtWidgets.QGroupBox("圆形 / 弦图", self.group_path)
        self.group_circle.setGeometry(16, 102, 464, 70)
        self.label_radius = QtWidgets.QLabel("半径", self.group_circle)
        self.label_radius.setGeometry(12, 28, 42, 20)
        self.edit_radius_c = QtWidgets.QLineEdit("90", self.group_circle)
        self.edit_radius_c.setGeometry(58, 28, 70, 28)
        self.label_nchord = QtWidgets.QLabel("弦数", self.group_circle)
        self.label_nchord.setGeometry(148, 31, 42, 20)
        self.edit_nchord = QtWidgets.QLineEdit("25", self.group_circle)
        self.edit_nchord.setGeometry(190, 28, 70, 28)
        self.group_circle.hide()

        self.group_line = QtWidgets.QGroupBox("直线", self.group_path)
        self.group_line.setGeometry(16, 102, 464, 70)
        self.label_line_x1 = QtWidgets.QLabel("起点", self.group_line)
        self.label_line_x1.setGeometry(12, 28, 42, 20)
        self.edit_line_x1 = QtWidgets.QLineEdit("90", self.group_line)
        self.edit_line_x1.setGeometry(56, 28, 54, 28)
        self.label_line_y1 = QtWidgets.QLabel(",", self.group_line)
        self.label_line_y1.setGeometry(114, 31, 10, 20)
        self.edit_line_y1 = QtWidgets.QLineEdit("90", self.group_line)
        self.edit_line_y1.setGeometry(126, 28, 54, 28)
        self.label_line_x2 = QtWidgets.QLabel("终点", self.group_line)
        self.label_line_x2.setGeometry(198, 31, 42, 20)
        self.edit_line_x2 = QtWidgets.QLineEdit("120", self.group_line)
        self.edit_line_x2.setGeometry(240, 28, 54, 28)
        self.label_line_y2 = QtWidgets.QLabel(",", self.group_line)
        self.label_line_y2.setGeometry(298, 31, 10, 20)
        self.edit_line_y2 = QtWidgets.QLineEdit("120", self.group_line)
        self.edit_line_y2.setGeometry(310, 28, 54, 28)
        self.label_line_step = QtWidgets.QLabel("步长", self.group_line)
        self.label_line_step.setGeometry(382, 31, 36, 20)
        self.edit_line_step = QtWidgets.QLineEdit("1.0", self.group_line)
        self.edit_line_step.setGeometry(418, 28, 36, 28)
        self.group_line.hide()

        self.label_workz = QtWidgets.QLabel("工作高度Z", self.group_path)
        self.label_workz.setGeometry(16, 190, 76, 22)
        self.edit_workz = QtWidgets.QLineEdit("5", self.group_path)
        self.edit_workz.setGeometry(92, 188, 60, 28)
        self.label_workz_unit = QtWidgets.QLabel("mm", self.group_path)
        self.label_workz_unit.setGeometry(160, 191, 28, 20)

        self.cb_3d = QtWidgets.QCheckBox("3D多层打印", self.group_path)
        self.cb_3d.setGeometry(222, 190, 112, 24)
        self.label_layers = QtWidgets.QLabel("层数", self.group_path)
        self.label_layers.setGeometry(348, 190, 36, 22)
        self.edit_layers = QtWidgets.QLineEdit("3", self.group_path)
        self.edit_layers.setGeometry(384, 188, 42, 28)
        self.edit_layers.setEnabled(False)
        self.label_layer_h = QtWidgets.QLabel("层高", self.group_path)
        self.label_layer_h.setGeometry(16, 236, 42, 22)
        self.edit_layer_h = QtWidgets.QLineEdit("1.0", self.group_path)
        self.edit_layer_h.setGeometry(58, 234, 60, 28)
        self.edit_layer_h.setEnabled(False)
        self.label_layer_h_unit = QtWidgets.QLabel("mm", self.group_path)
        self.label_layer_h_unit.setGeometry(126, 237, 28, 20)

        self.pb_preview = QtWidgets.QPushButton("预览轨迹", self.group_path)
        self.pb_preview.setGeometry(364, 230, 108, 34)
        self.pb_preview.setFont(font_button)
        self.label_traj_info = QtWidgets.QLabel("轨迹长度: -- mm    点数: --", self.group_path)
        self.label_traj_info.setGeometry(166, 236, 184, 24)
        self.label_traj_info.setStyleSheet("color:#607089;")

        self.group_print = QtWidgets.QGroupBox(self.centralwidget)
        self.group_print.setGeometry(QtCore.QRect(14, 766, 500, 74))
        self.group_print.setFont(font_group)
        self.group_print.setTitle("打印控制")

        self.pb_start_print = QtWidgets.QPushButton("开始打印", self.group_print)
        self.pb_start_print.setGeometry(24, 26, 210, 36)
        self.pb_start_print.setFont(font_button)
        self.pb_start_print.setStyleSheet("""
            QPushButton {
                background:#1f9d61;
                border-color:#1f9d61;
                color:white;
            }
            QPushButton:hover { background:#188b54; }
            QPushButton:pressed { background:#137345; }
        """)

        self.pb_abort = QtWidgets.QPushButton("紧急停止 (Esc)", self.group_print)
        self.pb_abort.setGeometry(264, 26, 210, 36)
        self.pb_abort.setFont(font_button)
        self.pb_abort.setStyleSheet("""
            QPushButton {
                background:#d93f3f;
                border-color:#d93f3f;
                color:white;
            }
            QPushButton:hover { background:#bf3030; }
            QPushButton:pressed { background:#9e2727; }
        """)

        # ==================== 右侧预览与日志 ====================
        self.group_preview = QtWidgets.QGroupBox(self.centralwidget)
        self.group_preview.setGeometry(QtCore.QRect(530, 80, 676, 498))
        self.group_preview.setFont(font_group)
        self.group_preview.setTitle("轨迹预览")

        self.widget_plot = QtWidgets.QWidget(self.group_preview)
        self.widget_plot.setGeometry(QtCore.QRect(12, 28, 652, 457))
        self.widget_plot.setObjectName("widget_plot")
        self.widget_plot.setStyleSheet("background:#ffffff;border-radius:6px;")

        self.group_log = QtWidgets.QGroupBox(self.centralwidget)
        self.group_log.setGeometry(QtCore.QRect(530, 590, 676, 214))
        self.group_log.setFont(font_group)
        self.group_log.setTitle("运行日志")
        self.text_log = QtWidgets.QTextEdit(self.group_log)
        self.text_log.setGeometry(12, 28, 652, 172)
        self.text_log.setReadOnly(True)
        self.text_log.setFont(font_mono)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
