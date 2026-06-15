  # -*- coding: utf-8 -*-
"""
3D打印综合控制台 — UI 界面定义
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PrintWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("PrintWindow")
        MainWindow.resize(960, 730)
        MainWindow.setWindowTitle("3D打印综合控制台")
        font_main = QtGui.QFont()
        font_main.setFamily("微软雅黑")
        font_main.setPointSize(9)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setFont(font_main)
        self.centralwidget.setObjectName("centralwidget")

        # ==================== 机械臂设置 ====================
        self.group_arm = QtWidgets.QGroupBox(self.centralwidget)
        self.group_arm.setGeometry(QtCore.QRect(15, 10, 430, 110))
        f = QtGui.QFont("微软雅黑", 9)
        self.group_arm.setFont(f)
        self.group_arm.setTitle("机械臂设置")

        # IP
        self.label_ip = QtWidgets.QLabel("IP地址", self.group_arm)
        self.label_ip.setGeometry(10, 25, 50, 20)
        self.edit_ip = QtWidgets.QLineEdit("169.254.45.1", self.group_arm)
        self.edit_ip.setGeometry(60, 23, 120, 24)

        # 速度
        self.label_speed = QtWidgets.QLabel("速度(mm/s)", self.group_arm)
        self.label_speed.setGeometry(195, 25, 70, 20)
        self.edit_speed = QtWidgets.QLineEdit("10", self.group_arm)
        self.edit_speed.setGeometry(265, 23, 50, 24)

        # 加速度
        self.label_accel = QtWidgets.QLabel("加速度(mm/s²)", self.group_arm)
        self.label_accel.setGeometry(325, 25, 90, 20)
        self.edit_accel = QtWidgets.QLineEdit("1000", self.group_arm)
        self.edit_accel.setGeometry(395, 23, 50, 24)

        # 交融半径
        self.label_blend = QtWidgets.QLabel("交融半径(m)", self.group_arm)
        self.label_blend.setGeometry(10, 55, 80, 20)
        self.edit_blend = QtWidgets.QLineEdit("0.002", self.group_arm)
        self.edit_blend.setGeometry(90, 53, 60, 24)

        # 连接/断开按钮
        self.pb_arm_connect = QtWidgets.QPushButton("连接机械臂", self.group_arm)
        self.pb_arm_connect.setGeometry(180, 53, 90, 26)
        self.lamp_arm = QtWidgets.QLabel(self.group_arm)
        self.lamp_arm.setGeometry(280, 58, 14, 14)
        self.lamp_arm.setStyleSheet("background-color:gray;border-radius:7px")
        self.label_arm_state = QtWidgets.QLabel("未连接", self.group_arm)
        self.label_arm_state.setGeometry(300, 55, 60, 20)

        # 抬高按钮
        self.pb_lift = QtWidgets.QPushButton("⬆ 抬高", self.group_arm)
        self.pb_lift.setGeometry(355, 50, 65, 28)
        self.pb_lift.setToolTip("移动机械臂到安全高度 (100, 100, 50)")

        # TCP 位置显示
        self.label_tcp = QtWidgets.QLabel("TCP: X=— Y=— Z=—", self.group_arm)
        self.label_tcp.setGeometry(10, 82, 350, 20)
        font_small = QtGui.QFont("Consolas", 8)
        self.label_tcp.setFont(font_small)

        # ==================== 喷头设置 ====================
        self.group_ext = QtWidgets.QGroupBox(self.centralwidget)
        self.group_ext.setGeometry(QtCore.QRect(15, 130, 430, 170))
        self.group_ext.setFont(f)
        self.group_ext.setTitle("喷头设置 (步进电机挤出机)")

        # 串口
        self.label_port = QtWidgets.QLabel("串口号", self.group_ext)
        self.label_port.setGeometry(10, 25, 45, 20)
        self.combo_port = QtWidgets.QComboBox(self.group_ext)
        self.combo_port.setGeometry(55, 23, 80, 24)

        # 波特率
        self.label_baud = QtWidgets.QLabel("波特率", self.group_ext)
        self.label_baud.setGeometry(150, 25, 45, 20)
        self.combo_baud = QtWidgets.QComboBox(self.group_ext)
        self.combo_baud.setGeometry(195, 23, 80, 24)
        bauds = ["9600", "14400", "19200", "38400", "56000", "57600", "115200"]
        self.combo_baud.addItems(bauds)
        self.combo_baud.setCurrentText("115200")

        # 连接按钮
        self.pb_ext_connect = QtWidgets.QPushButton("连接喷头", self.group_ext)
        self.pb_ext_connect.setGeometry(295, 21, 90, 26)
        self.lamp_ext = QtWidgets.QLabel(self.group_ext)
        self.lamp_ext.setGeometry(395, 26, 14, 14)
        self.lamp_ext.setStyleSheet("background-color:gray;border-radius:7px")
        self.label_ext_state = QtWidgets.QLabel("未连接", self.group_ext)
        self.label_ext_state.setGeometry(415, 23, 60, 20)

        # 挤出速度
        self.label_ext_rpm = QtWidgets.QLabel("挤出速度(r/min)", self.group_ext)
        self.label_ext_rpm.setGeometry(10, 58, 100, 20)
        self.edit_ext_rpm = QtWidgets.QLineEdit("1", self.group_ext)
        self.edit_ext_rpm.setGeometry(110, 56, 55, 24)

        # 挤出延迟
        self.label_delay = QtWidgets.QLabel("启动延迟(s)", self.group_ext)
        self.label_delay.setGeometry(180, 58, 80, 20)
        self.edit_delay = QtWidgets.QLineEdit("3", self.group_ext)
        self.edit_delay.setGeometry(260, 56, 50, 24)

        # 回抽脉冲
        self.label_retract = QtWidgets.QLabel("回抽脉冲", self.group_ext)
        self.label_retract.setGeometry(325, 58, 60, 20)
        self.edit_retract = QtWidgets.QLineEdit("2000", self.group_ext)
        self.edit_retract.setGeometry(385, 56, 55, 24)

        # 预挤出脉冲
        self.label_prime = QtWidgets.QLabel("预挤出脉冲", self.group_ext)
        self.label_prime.setGeometry(10, 90, 80, 20)
        self.edit_prime = QtWidgets.QLineEdit("2000", self.group_ext)
        self.edit_prime.setGeometry(90, 88, 55, 24)

        # 喷头控制按钮
        self.pb_ext_start = QtWidgets.QPushButton("启动挤出", self.group_ext)
        self.pb_ext_start.setGeometry(180, 87, 80, 26)
        self.pb_ext_stop = QtWidgets.QPushButton("停止挤出", self.group_ext)
        self.pb_ext_stop.setGeometry(270, 87, 80, 26)
        self.pb_ext_retract = QtWidgets.QPushButton("回抽", self.group_ext)
        self.pb_ext_retract.setGeometry(360, 87, 60, 26)

        # 预停距离
        self.label_prestop = QtWidgets.QLabel("预停距离(mm)", self.group_ext)
        self.label_prestop.setGeometry(10, 118, 90, 20)
        self.edit_prestop = QtWidgets.QLineEdit("0", self.group_ext)
        self.edit_prestop.setGeometry(105, 116, 50, 24)
        self.edit_prestop.setToolTip("路径剩余多少mm时提前停止挤出，0=不启用")

        # 挤出头实时状态
        self.label_ext_pos = QtWidgets.QLabel("位置:— 速度:— r/min", self.group_ext)
        self.label_ext_pos.setGeometry(10, 140, 350, 20)
        self.label_ext_pos.setFont(font_small)

        # ==================== 轨迹设置 ====================
        self.group_path = QtWidgets.QGroupBox(self.centralwidget)
        self.group_path.setGeometry(QtCore.QRect(15, 310, 490, 220))
        self.group_path.setFont(f)
        self.group_path.setTitle("轨迹参数")

        # 轨迹类型
        self.label_type = QtWidgets.QLabel("轨迹类型", self.group_path)
        self.label_type.setGeometry(10, 25, 60, 20)
        self.combo_type = QtWidgets.QComboBox(self.group_path)
        self.combo_type.setGeometry(70, 23, 100, 24)
        self.combo_type.addItems(["矩形填充", "矩形轮廓", "圆形", "弦图", "直线"])
        self.combo_type.setCurrentIndex(0)

        # 中心点
        self.label_center = QtWidgets.QLabel("中心点 (X,Y)", self.group_path)
        self.label_center.setGeometry(190, 25, 80, 20)
        self.edit_cx = QtWidgets.QLineEdit("100", self.group_path)
        self.edit_cx.setGeometry(275, 23, 45, 24)
        self.edit_cy = QtWidgets.QLineEdit("100", self.group_path)
        self.edit_cy.setGeometry(325, 23, 45, 24)

        # 矩形参数
        self.group_rect = QtWidgets.QGroupBox("矩形参数", self.group_path)
        self.group_rect.setGeometry(10, 55, 410, 60)
        self.label_w = QtWidgets.QLabel("宽(mm)", self.group_rect)
        self.label_w.setGeometry(8, 25, 50, 20)
        self.edit_width = QtWidgets.QLineEdit("10", self.group_rect)
        self.edit_width.setGeometry(60, 23, 50, 24)
        self.label_h = QtWidgets.QLabel("高(mm)", self.group_rect)
        self.label_h.setGeometry(125, 25, 45, 20)
        self.edit_height = QtWidgets.QLineEdit("10", self.group_rect)
        self.edit_height.setGeometry(175, 23, 50, 24)
        self.label_lw = QtWidgets.QLabel("线宽(mm)", self.group_rect)
        self.label_lw.setGeometry(240, 25, 60, 20)
        self.edit_linewidth = QtWidgets.QLineEdit("1.0", self.group_rect)
        self.edit_linewidth.setGeometry(305, 23, 45, 24)
        self.label_corner = QtWidgets.QLabel("起始角", self.group_rect)
        self.label_corner.setGeometry(360, 25, 45, 20)
        self.combo_corner = QtWidgets.QComboBox(self.group_rect)
        self.combo_corner.setGeometry(355, 23, 60, 24)
        self.combo_corner.addItems(["bl", "br", "tl", "tr"])

        # 圆形参数（半径 + 弦数）
        self.group_circle = QtWidgets.QGroupBox("圆形参数", self.group_path)
        self.group_circle.setGeometry(10, 55, 410, 60)
        self.label_radius = QtWidgets.QLabel("半径(mm)", self.group_circle)
        self.label_radius.setGeometry(8, 25, 60, 20)
        self.edit_radius_c = QtWidgets.QLineEdit("90", self.group_circle)
        self.edit_radius_c.setGeometry(70, 23, 50, 24)
        self.label_nchord = QtWidgets.QLabel("弦数", self.group_circle)
        self.label_nchord.setGeometry(140, 25, 35, 20)
        self.edit_nchord = QtWidgets.QLineEdit("25", self.group_circle)
        self.edit_nchord.setGeometry(175, 23, 45, 24)
        self.group_circle.hide()  # 默认矩形可见

        # 直线参数（起点, 终点, 步长）
        self.group_line = QtWidgets.QGroupBox("直线参数", self.group_path)
        self.group_line.setGeometry(10, 55, 410, 60)
        self.label_line_x1 = QtWidgets.QLabel("起点X", self.group_line)
        self.label_line_x1.setGeometry(8, 10, 40, 20)
        self.edit_line_x1 = QtWidgets.QLineEdit("90", self.group_line)
        self.edit_line_x1.setGeometry(45, 8, 45, 24)
        self.label_line_y1 = QtWidgets.QLabel("起点Y", self.group_line)
        self.label_line_y1.setGeometry(95, 10, 40, 20)
        self.edit_line_y1 = QtWidgets.QLineEdit("90", self.group_line)
        self.edit_line_y1.setGeometry(130, 8, 45, 24)
        self.label_line_x2 = QtWidgets.QLabel("终点X", self.group_line)
        self.label_line_x2.setGeometry(185, 10, 40, 20)
        self.edit_line_x2 = QtWidgets.QLineEdit("110", self.group_line)
        self.edit_line_x2.setGeometry(225, 8, 45, 24)
        self.label_line_y2 = QtWidgets.QLabel("终点Y", self.group_line)
        self.label_line_y2.setGeometry(275, 10, 40, 20)
        self.edit_line_y2 = QtWidgets.QLineEdit("110", self.group_line)
        self.edit_line_y2.setGeometry(315, 8, 45, 24)
        self.label_line_step = QtWidgets.QLabel("步长(mm)", self.group_line)
        self.label_line_step.setGeometry(8, 35, 60, 20)
        self.edit_line_step = QtWidgets.QLineEdit("1.0", self.group_line)
        self.edit_line_step.setGeometry(70, 33, 45, 24)
        self.group_line.hide()

        # 公共：工作高度
        self.label_workz = QtWidgets.QLabel("工作高度Z(mm)", self.group_path)
        self.label_workz.setGeometry(10, 125, 100, 20)
        self.edit_workz = QtWidgets.QLineEdit("5", self.group_path)
        self.edit_workz.setGeometry(115, 123, 50, 24)

        # 3D 多层打印
        self.cb_3d = QtWidgets.QCheckBox("3D多层打印", self.group_path)
        self.cb_3d.setGeometry(190, 123, 100, 22)
        self.label_layers = QtWidgets.QLabel("层数", self.group_path)
        self.label_layers.setGeometry(295, 125, 35, 20)
        self.edit_layers = QtWidgets.QLineEdit("3", self.group_path)
        self.edit_layers.setGeometry(330, 123, 40, 24)
        self.edit_layers.setEnabled(False)
        self.label_layer_h = QtWidgets.QLabel("层高(mm)", self.group_path)
        self.label_layer_h.setGeometry(380, 125, 60, 20)
        self.edit_layer_h = QtWidgets.QLineEdit("1.0", self.group_path)
        self.edit_layer_h.setGeometry(440, 123, 40, 24)
        self.edit_layer_h.setEnabled(False)

        # 预览按钮
        self.pb_preview = QtWidgets.QPushButton("预览轨迹", self.group_path)
        self.pb_preview.setGeometry(10, 165, 100, 30)
        font_btn = QtGui.QFont("微软雅黑", 9, QtGui.QFont.Bold)
        self.pb_preview.setFont(font_btn)

        # 轨迹信息
        self.label_traj_info = QtWidgets.QLabel("轨迹长度: — mm  点数: —", self.group_path)
        self.label_traj_info.setGeometry(130, 165, 300, 30)

        # ==================== 打印控制 ====================
        self.group_print = QtWidgets.QGroupBox(self.centralwidget)
        self.group_print.setGeometry(QtCore.QRect(15, 540, 430, 80))
        self.group_print.setFont(f)
        self.group_print.setTitle("打印控制")

        self.pb_start_print = QtWidgets.QPushButton("▶  开始打印", self.group_print)
        self.pb_start_print.setGeometry(20, 25, 130, 40)
        self.pb_start_print.setStyleSheet(
            "QPushButton{background-color:#4CAF50;color:white;font-size:14px;font-weight:bold;"
            "border-radius:6px}QPushButton:hover{background-color:#45a049}")

        self.pb_abort = QtWidgets.QPushButton("■  紧急停止", self.group_print)
        self.pb_abort.setGeometry(170, 25, 130, 40)
        self.pb_abort.setStyleSheet(
            "QPushButton{background-color:#f44336;color:white;font-size:14px;font-weight:bold;"
            "border-radius:6px}QPushButton:hover{background-color:#da190b}")

        self.label_print_state = QtWidgets.QLabel("就绪", self.group_print)
        self.label_print_state.setGeometry(320, 30, 100, 30)
        font_state = QtGui.QFont("微软雅黑", 12, QtGui.QFont.Bold)
        self.label_print_state.setFont(font_state)

        # ==================== 日志区域 ====================
        self.group_log = QtWidgets.QGroupBox(self.centralwidget)
        self.group_log.setGeometry(QtCore.QRect(15, 630, 430, 80))
        self.group_log.setFont(f)
        self.group_log.setTitle("运行日志")
        self.text_log = QtWidgets.QTextEdit(self.group_log)
        self.text_log.setGeometry(5, 18, 420, 56)
        self.text_log.setReadOnly(True)
        self.text_log.setFont(QtGui.QFont("Consolas", 8))

        # ==================== 路径预览图 ====================
        self.widget_plot = QtWidgets.QWidget(self.centralwidget)
        self.widget_plot.setGeometry(QtCore.QRect(520, 10, 425, 610))
        self.widget_plot.setObjectName("widget_plot")
        # 实际 canvas 在 window 中创建

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)