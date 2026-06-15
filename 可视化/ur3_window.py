import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QTimer
import threading
from ur3_ui import Ui_MainWindow
from UR3Controller import UR3Controller
# from shapely import Polygon
# from contour import Contour, find_contour
# from curve_optimize import Spiral_Curve


points = [[0, -100, 0], [0, -100, 100], [0, -80, 100], [0, -80, 20], [0, -60, 20],  [0, -60, 100],
          [0, -40, 100], [0, -40, 20], [0, -20, 20], [0, -20, 100], [0, 0, 100], [0, 0, 0]]


def scaling(points, scale):
    new_points = []
    for point in points:
        new_points.append([pt * scale for pt in point])
    return new_points


def try_parse_float(text):
    try:
        return float(text)
    except ValueError:
        return 0


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 创建一个Figure对象
        self.figure, self.axes = plt.subplots()
        self.canvas = FigureCanvas(self.figure)  # 创建FigureCanvas
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.widget_plot.setLayout(QVBoxLayout())
        self.widget_plot.layout().addWidget(self.canvas)  # 将FigureCanvas添加到PlotWidget

        self.pb_connect.clicked.connect(self.ur3_connect)
        self.lamp_ip_state.setStyleSheet('background-color: red; border-radius: 8px')
        self.lamp_ur3_state.setStyleSheet('background-color: gray; border-radius: 8px')
        self.pb_set_zero.clicked.connect(self.set_zero)
        self.pb_move_to_zero.clicked.connect(self.move_to_zero)
        # self.pb_load_boundary.clicked.connect(self.load_boundary)
        self.pb_sprial_fill.clicked.connect(self.spiral_fill)
        # self.pb_curve_optimize.clicked.connect(self.curve_optimize)
        self.pb_ur3_move.clicked.connect(self.move_path)

        self.pb_set_params.clicked.connect(self.set_params)
        self.pb_move_by_current.clicked.connect(self.move_by_current)
        self.pb_move_by_zero.clicked.connect(self.move_by_zero)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout_callback)

        self.tcp_pose = []
        self.jogs = []
        self.xyz = []
        self.speed = 0
        self.ur3 = None
        self.time = 0

        self.file = 'speed.txt'

    def ur3_connect(self):
        ip_address = self.lineEdit_ip_address.text()
        try:
            self.ur3 = UR3Controller(ip_address)
            self.lineEdit_speed.setText(f'{self.ur3.speed}')
            self.lineEdit_acceleration.setText(f'{self.ur3.acceleration}')
            self.lineEdit_blend.setText(f'{self.ur3.blend}')
            self.tcp_pose = self.ur3.tcp_pose

            self.lamp_ip_state.setStyleSheet('background-color: rgb(102, 255, 0); border-radius: 8px')
            self.pb_connect.setText('断开')
            self.lamp_ur3_state.setStyleSheet('background-color: red; border-radius: 8px')
            self.label_ur3_state.setText('空闲中')
            self.timer.start(20)
        except Exception as e:
            print(e)

    def plot(self, points, is_closed=True):
        # 在Figure对象中创建图形
        x = [pt[0] for pt in points]
        y = [pt[1] for pt in points]
        if is_closed:
            x.append(x[0])
            y.append(y[0])
        self.axes.plot(x, y)
        # 更新图形
        self.canvas.draw()
        # self.canvas.draw_idle()
        # self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.widget_plot.updateGeometry()

    def set_zero(self):
        if self.ur3:
            self.ur3.set_zero()

    def move_to_zero(self):
        if self.ur3:
            self.ur3.move_to_zero()

    # def load_boundary(self):
    #     path = r'images\irregular.png'
    #     contours = find_contour(path)
    #     edge = contours[0]
    #     new_edge = scaling(edge, 1/10)
    #     self.axes.clear()
    #     self.plot(new_edge)
    #     inters = contours[1:]
    #     new_inters = []
    #     for inter in inters:
    #         inter = scaling(inter, 1/10)
    #         self.plot(inter)
    #         new_inters.append(inter)

    #     polygon = Polygon(new_edge, new_inters)
    #     self.contour = Contour(polygon)

    def spiral_fill(self):
        self.contour.fill(-2.3)
        self.contour.connect(2.3)
        self.axes.clear()
        self.plot(self.contour.spiral, False)

    # def curve_optimize(self):
    #     self.spiral = Spiral_Curve(self.contour.spiral)
    #     self.spiral.nonuniform_sampling(2.3 / 4)
    #     self.spiral.gradient_descent(space=2.3, iterations=100, learning_rate=0.02)
    #     self.axes.clear()
    #     self.plot(self.spiral.coords, False)

    def move_path(self):
        if self.ur3:
            path = [[pt[0], pt[1], 0] for pt in self.spiral.coords]
            self.ur3.move_path(path, current=True)
            self.axes.clear()
            self.xyz = []

    def set_params(self):
        speed = float(self.lineEdit_speed.text())
        acceleration = float(self.lineEdit_acceleration.text())
        blend = float(self.lineEdit_blend.text())
        self.ur3.speed = speed
        self.ur3.acceleration = acceleration
        self.ur3.blend = blend

    def move_by_current(self):
        if self.ur3:
            x = try_parse_float(self.lineEdit_move_x.text())
            y = try_parse_float(self.lineEdit_move_y.text())
            z = try_parse_float(self.lineEdit_move_z.text())
            self.ur3.move_to_target([x, y, z], current=True)

    def move_by_zero(self):
        if self.ur3:
            x = try_parse_float(self.lineEdit_move_x.text())
            y = try_parse_float(self.lineEdit_move_y.text())
            z = try_parse_float(self.lineEdit_move_z.text())
            self.ur3.move_to_target([x, y, z], current=False)

    def update_plot(self):
        if len(self.xyz) == 2:
            x = [pt[0] for pt in self.xyz]
            y = [pt[1] for pt in self.xyz]
            self.line = self.axes.plot(x, y)[0]
            self.axes.set_title('Real-time Data')
        elif len(self.xyz) > 2:
            x = [pt[0] for pt in self.xyz]
            y = [pt[1] for pt in self.xyz]
            self.line.set_xdata(x)
            self.line.set_ydata(y)
            xmin, xmax = min(x), max(x)
            ymin, ymax = min(y), max(y)
            self.axes.set_xlim([xmin - 0.01, xmax + 0.01])
            self.axes.set_ylim([ymin - 0.01, ymax + 0.01])
            self.canvas.draw()
            self.canvas.flush_events()

    def update_ui(self):
        x, y, z, rx, ry, rz = self.tcp_pose
        x, y, z = x * 1000, y * 1000, z * 1000
        jog1, jog2, jog3, jog4, jog5, jog6 = [jog * 180 / np.pi for jog in self.jogs]
        self.lineEdit_tcp_x.setText(f'{x:.2f}')
        self.lineEdit_tcp_y.setText(f'{y:.2f}')
        self.lineEdit_tcp_z.setText(f'{z:.2f}')
        self.lineEdit_tcp_rx.setText(f'{rx:.2f}')
        self.lineEdit_tcp_ry.setText(f'{ry:.2f}')
        self.lineEdit_tcp_rz.setText(f'{rz:.2f}')
        self.lineEdit_tcp_speed.setText(f'{self.speed:.2f}')
        self.lineEdit_jog1.setText(f'{jog1:.2f}')
        self.lineEdit_jog2.setText(f'{jog2:.2f}')
        self.lineEdit_jog3.setText(f'{jog3:.2f}')
        self.lineEdit_jog4.setText(f'{jog4:.2f}')
        self.lineEdit_jog5.setText(f'{jog5:.2f}')
        self.lineEdit_jog6.setText(f'{jog6:.2f}')

        if not self.ur3.state:
            self.xyz.append(self.tcp_pose[0:3])
            self.update_plot()
            self.lamp_ur3_state.setStyleSheet('background-color: rgb(102, 255, 0); border-radius: 8px')
            self.label_ur3_state.setText('运行中')
        else:
            self.xyz = []
            self.lamp_ur3_state.setStyleSheet('background-color: red; border-radius: 8px')
            self.label_ur3_state.setText('空闲中')

    def timeout_callback(self):
        self.time += 0.05
        self.tcp_pose = self.ur3.receive.getActualTCPPose()
        velocity = self.ur3.receive.getActualTCPSpeed()
        vx, vy, vz, vrx, vry, vrz = velocity
        self.speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * 1000
        self.jogs = self.ur3.receive.getActualQ()
        self.ur3.state = self.ur3.control.isSteady()
        thread = threading.Thread(target=self.update_ui)
        thread.start()

    def stop(self):
        self.ur3.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
