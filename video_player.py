import sys
import cv2
import numpy as np
import os
import collections
import math

try:
    import onnxruntime as ort
except ImportError:
    print("Hata: onnxruntime kütüphanesi bulunamadı. Lütfen 'pip install onnxruntime' komutunu çalıştırın.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Hata: matplotlib kütüphanesi bulunamadı. Lütfen 'pip install matplotlib' komutunu çalıştırın.")
    sys.exit(1)

import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QStatusBar, QMenuBar, QAction, QGridLayout, QSizePolicy, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont

from settings_dialog import SettingsDialog
from calibration_dialog import CalibrationDialog

# Global sabitler
IMG_SIZE = 640
IOU_THRESHOLD = 0.45
CALIBRATION_FILE = "calibration_data.json"

# YOLOv8 için sınıf adları ve renkleri (Örnek olarak)
CLASS_NAMES = ["mouse", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# Fare sınıfı ID'si (Modelinizin çıktısına göre doğru ID'yi buraya girin)
MOUSE_CLASS_ID = 0 
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128)]


class VideoLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        scaled_pix = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        point = QPoint((self.width() - scaled_pix.width()) // 2, (self.height() - scaled_pix.height()) // 2)
        painter.drawPixmap(point, scaled_pix)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2E2E2E')
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')
        self.axes.tick_params(axis='z', colors='white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.zaxis.label.set_color('white')
        self.axes.title.set_color('lime')
        self.axes.set_facecolor('#2E2E2E')
        fig.tight_layout()
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class VideoPlayerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Çoklu Video Fare Tespiti ve 3D Konumlandırma")
        self.setGeometry(100, 100, 1600, 1000)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self._create_menu_bar()
        self._create_video_display_area(main_layout)
        self._create_control_bar(main_layout)
        self._create_status_bar()

        self.video_captures = [None] * 3
        self.onnx_sessions = [None] * 3
        self.onnx_paths = ["", "", ""]
        self.conf_thresholds = [0.25, 0.25, 0.25]
        self.video_paths = ["", "", ""]
        self.iou_threshold = IOU_THRESHOLD

        self.camera_matrices = [None] * 3
        self.dist_coeffs = [None] * 3
        self.R_01, self.T_01 = None, None
        self.R_02, self.T_02 = None, None
        self.is_calibrated = False

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.last_kalman_prediction = None
        self.point_history = collections.deque(maxlen=100)

        self.plot_elements = {'path': None, 'point': None, 'shadow_line': None}
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.is_playing = False

        self.load_calibration_data()
        self.update_button_states()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Ayarlar")
        settings_action = QAction("Giriş ve Model Ayarları...", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(settings_action)

        calibration_action = QAction("Kameraları Kalibre Et...", self)
        calibration_action.triggered.connect(self.open_calibration_dialog)
        settings_menu.addAction(calibration_action)

        exit_menu = menu_bar.addMenu("Çıkış")
        exit_action = QAction("Çıkış", self)
        exit_action.triggered.connect(self.close)
        exit_menu.addAction(exit_action)

    def _create_video_display_area(self, layout):
        video_grid_layout = QGridLayout()
        video_grid_layout.setSpacing(10)

        self.video_areas = []
        for i in range(3):
            video_area = VideoLabel(f"Video {i+1} bekleniyor...")
            self.video_areas.append(video_area)
            video_grid_layout.addWidget(video_area, 0, i)

        # 3D görselleştirme için bir QWidget container'ı oluştur
        three_d_container_widget = QWidget()
        three_d_container_layout = QVBoxLayout(three_d_container_widget)
        three_d_container_widget.setStyleSheet("background-color: #202020; border: 2px solid gray; padding: 5px;")

        # 3D Konum metnini gösterecek QLabel
        self.three_d_view_label = QLabel("3D Konum Görselleştirme\n\nFare tespiti bekleniyor...")
        self.three_d_view_label.setAlignment(Qt.AlignCenter)
        self.three_d_view_label.setStyleSheet("color: #00FF00; font-size: 12pt;")
        self.three_d_view_label.setWordWrap(True)
        three_d_container_layout.addWidget(self.three_d_view_label)

        # Matplotlib canvas'ı
        self.three_d_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.three_d_canvas.axes.set_xlabel('X (Genişlik)')
        self.three_d_canvas.axes.set_ylabel('Y (Derinlik)')
        self.three_d_canvas.axes.set_zlabel('Z (Yükseklik)')
        self.three_d_canvas.axes.set_title("Fare Hareket İzi")
        self.three_d_canvas.axes.set_ylim([-50, 50])
        self.three_d_canvas.axes.set_xlim([0, 1280])
        self.three_d_canvas.axes.set_zlim([0, 720])
        self.three_d_canvas.axes.invert_zaxis()
        
        xx, yy = np.meshgrid(np.linspace(0, 1280, 5), np.linspace(-50, 50, 5))
        zz = np.zeros_like(xx)
        self.three_d_canvas.axes.plot_wireframe(xx, yy, zz, color="gray", alpha=0.5)
        
        three_d_container_layout.addWidget(self.three_d_canvas)
        
        video_grid_layout.addWidget(three_d_container_widget, 1, 0, 1, 3)

        layout.addLayout(video_grid_layout)

    def _create_control_bar(self, layout):
        control_bar_layout = QHBoxLayout()
        control_bar_layout.setSpacing(10)

        self.play_button = QPushButton("▶")
        self.play_button.clicked.connect(self.toggle_play_pause)
        self.play_button.setFixedSize(50, 50)
        self.play_button.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.rewind_button = QPushButton("⏮")
        self.rewind_button.clicked.connect(self.rewind_videos)
        self.rewind_button.setFixedSize(50, 50)
        self.rewind_button.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.forward_button = QPushButton("⏭")
        self.forward_button.clicked.connect(self.forward_videos)
        self.forward_button.setFixedSize(50, 50)
        self.forward_button.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.sliderMoved.connect(self.set_video_position)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(120)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px;")

        control_bar_layout.addStretch()
        control_bar_layout.addWidget(self.rewind_button)
        control_bar_layout.addWidget(self.play_button)
        control_bar_layout.addWidget(self.forward_button)
        control_bar_layout.addStretch()
        control_bar_layout.addWidget(self.progress_slider, 1)
        control_bar_layout.addWidget(self.time_label)
        control_bar_layout.addStretch()

        layout.addLayout(control_bar_layout)
    
    def _create_status_bar(self):
        self.statusBar().showMessage("Ayarlar menüsünden video ve model seçmeye hazır.")

    def open_settings_dialog(self):
        current_settings = {
            "input_paths": self.video_paths,
            "onnx_paths": self.onnx_paths,
            "conf_thresholds": self.conf_thresholds,
            "iou_threshold": self.iou_threshold
        }
        settings_dialog = SettingsDialog(self, initial_settings=current_settings)
        if settings_dialog.exec_() == QDialog.Accepted:
            settings = settings_dialog.get_settings()
            self.video_paths = settings["input_paths"]
            self.onnx_paths = settings["onnx_paths"]
            self.conf_thresholds = settings["conf_thresholds"]
            self.iou_threshold = settings["iou_threshold"]

            self.statusBar().showMessage("Ayarlar kaydedildi. Videolar ve modeller yükleniyor...")
            
            self.load_models()
            self.load_videos()
        else:
            self.statusBar().showMessage("Ayarlar iptal edildi.")

    def open_calibration_dialog(self):
        if not any(vc is not None and vc.isOpened() for vc in self.video_captures):
            QMessageBox.warning(self, "Uyarı", "Lütfen kalibrasyona başlamadan önce video kaynaklarını ayarlayıp yükleyin.")
            return
        calibration_dialog = CalibrationDialog(self, video_captures=self.video_captures)
        calibration_dialog.calibration_done_signal.connect(self.receive_calibration_data)
        calibration_dialog.exec_()

    def receive_calibration_data(self, matrices, dists, R_01, T_01, R_02, T_02):
        self.camera_matrices = matrices
        self.dist_coeffs = dists
        self.R_01 = R_01
        self.T_01 = T_01
        self.R_02 = R_02
        self.T_02 = T_02
        self.is_calibrated = True
        self.statusBar().showMessage("Kalibrasyon verileri başarıyla yüklendi.")
        self.save_calibration_data()

    def load_calibration_data(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE,'r') as f:
                    data = json.load(f)
                
                for i in range(3):
                    key = f'camera_{i}'
                    if key in data:
                        self.camera_matrices[i] = np.array(data[key]['camera_matrix'])
                        self.dist_coeffs[i] = np.array(data[key]['dist_coeffs'])

                if 'stereo_01' in data:
                    self.R_01 = np.array(data['stereo_01']['R'])
                    self.T_01 = np.array(data['stereo_01']['T'])
                
                if 'stereo_02' in data:
                    self.R_02 = np.array(data['stereo_02']['R'])
                    self.T_02 = np.array(data['stereo_02']['T'])
                
                self.is_calibrated = True # Kalibrasyon verileri yüklendiğinde True yap
                self.statusBar().showMessage("Kaydedilmiş kalibrasyon verileri yüklendi.")
                return
            except Exception as e:
                self.statusBar().showMessage(f"Kalibrasyon verileri yüklenemedi: {e}")
        
        self.statusBar().showMessage("UYARI: Sanal kamera ayarları kullanılıyor.")
        W, H = 1280, 720
        default_K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float32)
        default_D = np.zeros(5, dtype=np.float32)
        for i in range(3):
            self.camera_matrices[i] = default_K
            self.dist_coeffs[i] = default_D
        
        self.R_01 = np.eye(3, dtype=np.float32)
        self.T_01 = np.array([[30.0], [0.0], [0.0]], dtype=np.float32)
        
        angle = np.pi / 2
        self.R_02 = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        self.T_02 = np.array([[0.0], [30.0], [0.0]], dtype=np.float32)
        self.is_calibrated = False # Varsayılan kalibrasyonla gerçek kalibrasyon yapılmadı

    def save_calibration_data(self):
        calib_data = {}
        for i in range(3):
            if self.camera_matrices[i] is not None:
                calib_data[f'camera_{i}'] = {
                    'camera_matrix': self.camera_matrices[i].tolist(),
                    'dist_coeffs': self.dist_coeffs[i].tolist()
                }
        
        if self.R_01 is not None and self.T_01 is not None:
            calib_data['stereo_01'] = {
                'R': self.R_01.tolist(),
                'T': self.T_01.tolist()
            }
        
        if self.R_02 is not None and self.T_02 is not None:
            calib_data['stereo_02'] = {
                'R': self.R_02.tolist(),
                'T': self.T_02.tolist()
            }

        if not calib_data:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek kalibrasyon verisi bulunamadı.")
            return

        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(calib_data, f, indent=4)
            self.statusBar().showMessage(f"Kalibrasyon verileri {CALIBRATION_FILE} konumuna kaydedildi.")
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Kalibrasyon verileri kaydedilirken bir hata oluştu: {e}")

    def load_models(self):
        for i, path in enumerate(self.onnx_paths):
            if self.onnx_sessions[i]:
                del self.onnx_sessions[i]
            
            if path and os.path.exists(path):
                try:
                    providers_to_use = ["CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        providers_to_use.insert(0, "CUDAExecutionProvider")

                    self.onnx_sessions[i] = ort.InferenceSession(path, providers=providers_to_use)
                    self.statusBar().showMessage(f"Model {i+1} ({path}) başarıyla yüklendi. Sağlayıcılar: {self.onnx_sessions[i].get_providers()}")
                except Exception as e:
                    self.onnx_sessions[i] = None
                    QMessageBox.warning(self, "Model Hatası", f"Model {i+1} yüklenemedi: {e}")
            else:
                self.onnx_sessions[i] = None
                if path:
                    self.statusBar().showMessage(f"Hata: Model {i+1} için dosya bulunamadı: {path}")

    def load_videos(self):
        if self.is_playing:
            self.toggle_play_pause()

        for i, path in enumerate(self.video_paths):
            if self.video_captures[i]:
                self.video_captures[i].release()
            
            if not path:
                self.video_captures[i] = None
                self.video_areas[i].setText(f"Video {i+1} Kaynağı Yok")
                continue
            
            try:
                source = int(path)
                cap = cv2.VideoCapture(source)
            except ValueError:
                cap = cv2.VideoCapture(path)

            if cap.isOpened():
                self.video_captures[i] = cap
                self.video_areas[i].setText(f"Video {i+1} Akışı")
            else:
                self.video_captures[i] = None
                self.video_areas[i].setText(f"Video {i+1} Açılamadı")
                QMessageBox.warning(self, "Video Hatası", f"Video {i+1} ({path}) açılamadı.")
        
        self.update_button_states()
        self.point_history.clear()
        self.last_kalman_prediction = None

        if any(vc is not None and vc.isOpened() for vc in self.video_captures):
            self.statusBar().showMessage("Videolar yüklendi ve oynamaya hazır.")
        else:
            self.statusBar().showMessage("Hiçbir video kaynağı açılamadı.")

    def update_button_states(self):
        is_video_loaded = any(vc and vc.isOpened() for vc in self.video_captures)
        self.play_button.setEnabled(is_video_loaded)
        self.rewind_button.setEnabled(is_video_loaded)
        self.forward_button.setEnabled(is_video_loaded)
        self.progress_slider.setEnabled(is_video_loaded)

    def toggle_play_pause(self):
        if not any(vc is not None and vc.isOpened() for vc in self.video_captures):
            QMessageBox.warning(self, "Uyarı", "Lütfen oynatmaya başlamadan önce video kaynaklarını yükleyin.")
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.timer.start(30)
            self.play_button.setText("❚❚")
            self.statusBar().showMessage("Oynatılıyor...")
        else:
            self.timer.stop()
            self.play_button.setText("▶")
            self.statusBar().showMessage("Duraklatıldı.")

    def set_video_position(self, value):
        ref_cap = next((vc for vc in self.video_captures if vc and vc.isOpened()), None)
        if ref_cap and ref_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            total_frames = ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_to_set = int(total_frames * (value / 100))
            for vc in self.video_captures:
                if vc and vc.isOpened():
                    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_to_set)
            if not self.is_playing:
                self.update_frames()
        self.update_slider_and_time()

    def rewind_videos(self):
        ref_cap = next((vc for vc in self.video_captures if vc and vc.isOpened()), None)
        if ref_cap:
            fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30
            current_frame = ref_cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_frame = max(0, current_frame - fps * 5)
            for vc in self.video_captures:
                if vc and vc.isOpened():
                    vc.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            if not self.is_playing:
                self.update_frames()
        self.update_slider_and_time()

    def forward_videos(self):
        ref_cap = next((vc for vc in self.video_captures if vc and vc.isOpened()), None)
        if ref_cap:
            fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30
            current_frame = ref_cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            new_frame = min(total_frames - 1, current_frame + fps * 5)
            for vc in self.video_captures:
                if vc and vc.isOpened():
                    vc.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            if not self.is_playing:
                self.update_frames()
        self.update_slider_and_time()
    
    def update_frames(self):
        all_videos_ended = True
        self.mouse_3d_coordinates = []
        detections_per_camera = [[] for _ in range(3)]

        for i in range(3):
            vc = self.video_captures[i]
            if vc and vc.isOpened():
                all_videos_ended = False
                ret, frame = vc.read()
                if ret:
                    display_frame = frame.copy()
                    h, w, _ = display_frame.shape

                    if self.camera_matrices[i] is not None and self.dist_coeffs[i] is not None:
                        display_frame = cv2.undistort(display_frame, self.camera_matrices[i], self.dist_coeffs[i], None, self.camera_matrices[i])

                    if self.onnx_sessions[i]:
                        try:
                            img_input, scale_factor, padding_offset = self._preprocess_frame_for_onnx(display_frame)

                            ort_inputs = {self.onnx_sessions[i].get_inputs()[0].name: img_input}
                            ort_outs = self.onnx_sessions[i].run(None, ort_inputs)
                            detections = ort_outs[0]

                            current_detections = self.post_process_detections(detections, h, w, scale_factor, padding_offset, self.conf_thresholds[i], self.iou_threshold)
                            detections_per_camera[i] = current_detections

                            for det in current_detections:
                                x1, y1, x2, y2, score, class_id = det
                                
                                color = COLORS[int(class_id) % len(COLORS)]
                                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                label = f"{CLASS_NAMES[int(class_id)]}: {score:.2f}"
                                cv2.putText(display_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                        except Exception as e:
                            self.statusBar().showMessage(f"Model tahmin hatası (Kamera {i+1}): {e}")
                            print(f"Model tahmin hatası (Kamera {i+1}): {e}")

                    qt_img = self._convert_frame_to_pixmap(display_frame, self.video_areas[i])
                    self.video_areas[i].setPixmap(qt_img)
                else:
                    vc.release()
                    self.video_captures[i] = None
                    self.video_areas[i].setText(f"Video {i+1} sona erdi.")
            else:
                self.video_areas[i].setText(f"Video {i+1} yüklü değil/açık değil.")

        if self.is_calibrated and detections_per_camera[0] and detections_per_camera[1]:
            mice_cam0 = [d for d in detections_per_camera[0] if CLASS_NAMES[int(d[5])] == "mouse"]
            mice_cam1 = [d for d in detections_per_camera[1] if CLASS_NAMES[int(d[5])] == "mouse"]

            if mice_cam0 and mice_cam1:
                mouse0_det = mice_cam0[0]
                mouse1_det = mice_cam1[0]

                center0 = ((mouse0_det[0] + mouse0_det[2]) / 2, (mouse0_det[1] + mouse0_det[3]) / 2)
                center1 = ((mouse1_det[0] + mouse1_det[2]) / 2, (mouse1_det[1] + mouse1_det[3]) / 2)

                try:
                    P0 = self.camera_matrices[0] @ np.hstack((np.eye(3), np.zeros((3,1))))
                    P1 = self.camera_matrices[1] @ np.hstack((self.R_01, self.T_01))
                    
                    pt0_2d = np.array([center0[0], center0[1]], dtype=np.float32)
                    pt1_2d = np.array([center1[0], center1[1]], dtype=np.float32)

                    point3D = self.triangulate_points(pt0_2d, pt1_2d, 0, 1)
                    
                    if self.last_kalman_prediction is None:
                        self.kalman.statePost = np.array([[point3D[0]], [point3D[1]], [0.0], [0.0]], dtype=np.float32)
                    
                    measurement = np.array([[np.float32(point3D[0])], [np.float32(point3D[1])]])
                    self.kalman.correct(measurement)
                    filtered_point = self.kalman.statePost
                    self.last_kalman_prediction = filtered_point
                    
                    self.point_history.append(filtered_point[:3, 0])
                    
                except Exception as e:
                    print(f"3D Triangulasyon Hatası: {e}")
                    self.mouse_3d_coordinates.append(f"Hata: {e}")
            
        self.update_3d_view()
        self.update_slider_and_time()

        if all_videos_ended and self.is_playing:
            self.toggle_play_pause()


    def _preprocess_frame_for_onnx(self, frame):
        original_h, original_w, _ = frame.shape
        
        scale = IMG_SIZE / max(original_h, original_w)
        resized_w, resized_h = int(original_w * scale), int(original_h * scale)
        
        img_resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        
        dw = (IMG_SIZE - resized_w) / 2
        dh = (IMG_SIZE - resized_h) / 2
        
        img_padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        img_padded[int(dh):int(dh) + resized_h, int(dw):int(dw) + resized_w, :] = img_resized
        
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        return img_tensor, scale, (dw, dh)

    def post_process_detections(self, detections_output, original_h, original_w, scale_factor, padding_offset, conf_threshold, iou_threshold):
        detections = detections_output[0]
        detections = detections.T

        scores = detections[:, 4]
        class_scores = detections[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        final_scores = scores * np.max(class_scores, axis=1)

        conf_mask = final_scores > conf_threshold
        filtered_detections = detections[conf_mask]
        filtered_scores = final_scores[conf_mask]
        filtered_class_ids = class_ids[conf_mask]

        if filtered_detections.shape[0] == 0:
            return []

        boxes_raw = filtered_detections[:, :4]
        x_center, y_center, box_w, box_h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]

        dw, dh = padding_offset
        x1 = (x_center - box_w / 2 - dw) / scale_factor
        y1 = (y_center - box_h / 2 - dh) / scale_factor
        x2 = (x_center + box_w / 2 - dw) / scale_factor
        y2 = (y_center + box_h / 2 - dh) / scale_factor
        
        x1 = np.clip(x1, 0, original_w)
        y1 = np.clip(y1, 0, original_h)
        x2 = np.clip(x2, 0, original_w)
        y2 = np.clip(y2, 0, original_h)

        nms_boxes = np.vstack([x1, y1, x2 - x1, y2 - y1]).T.tolist()
        
        indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_scores.tolist(), conf_threshold, iou_threshold)

        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = nms_boxes[i]
                score = filtered_scores[i]
                class_id = filtered_class_ids[i]
                final_detections.append((box[0], box[1], box[0]+box[2], box[1]+box[3], score, class_id))
            
        return final_detections

    def triangulate_points(self, p1_2d, p2_2d, cam_idx1, cam_idx2):
        K1, D1 = self.camera_matrices[cam_idx1], self.dist_coeffs[cam_idx1]
        K2, D2 = self.camera_matrices[cam_idx2], self.dist_coeffs[cam_idx2]
        
        if cam_idx1 == 0 and cam_idx2 == 1:
            R, T = self.R_01, self.T_01
        elif cam_idx1 == 0 and cam_idx2 == 2:
            R, T = self.R_02, self.T_02
        else:
            raise ValueError("Desteklenmeyen kamera çifti veya kalibrasyon verisi eksik.")

        if any(v is None for v in [K1, D1, K2, D2, R, T]):
            raise ValueError("Kalibrasyon verileri eksik.")

        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K2 @ np.hstack((R, T))

        p1_u = cv2.undistortPoints(p1_2d.reshape(-1, 1, 2), K1, D1, None, K1)
        p2_u = cv2.undistortPoints(p2_2d.reshape(-1, 1, 2), K2, D2, None, K2)
        
        point_3d_homog = cv2.triangulatePoints(P1, P2, p1_u.reshape(2,1), p2_u.reshape(2,1))
        
        return (point_3d_homog / point_3d_homog[3])[:3].flatten()

    def update_3d_view(self):
        self.three_d_canvas.axes.cla()
        self.three_d_canvas.axes.set_xlabel('X (Genişlik)'); self.three_d_canvas.axes.set_ylabel('Y (Derinlik)'); self.three_d_canvas.axes.set_zlabel('Z (Yükseklik)')
        self.three_d_canvas.axes.set_title("Fare Hareket İzi", color='lime')
        self.three_d_canvas.axes.set_ylim([-50, 50]); self.three_d_canvas.axes.set_xlim([0, 1280]); self.three_d_canvas.axes.set_zlim([0, 720]); self.three_d_canvas.axes.invert_zaxis()
        
        xx, yy = np.meshgrid(np.linspace(0, 1280, 5), np.linspace(-50, 50, 5))
        zz = np.zeros_like(xx)
        self.three_d_canvas.axes.plot_wireframe(xx, yy, zz, color="gray", alpha=0.5)

        if not self.point_history:
            self.three_d_view_label.setText("3D Konum Görselleştirme\n\nFare tespiti bekleniyor...")
            self.three_d_canvas.draw()
            return

        points = np.array(self.point_history)
        
        self.plot_elements['path'], = self.three_d_canvas.axes.plot(points[:,0], points[:,1], points[:,2], color='cyan', alpha=0.7)
        
        last_point = self.point_history[-1]
        self.plot_elements['point'] = self.three_d_canvas.axes.scatter(last_point[0], last_point[1], last_point[2], c='lime', marker='o', s=100)
        
        self.plot_elements['shadow_line'], = self.three_d_canvas.axes.plot([last_point[0], last_point[0]], [last_point[1], last_point[1]], [last_point[2], 0], color='white', linestyle=':', alpha=0.6)
        
        self.three_d_canvas.axes.set_title(f"X:{last_point[0]:.0f}, Y:{last_point[1]:.0f}, Z:{last_point[2]:.0f}", color='lime')
        self.three_d_canvas.draw()

        text = "Tespit Edilen Fare 3D Koordinatları:\n\n"
        if self.point_history:
            last_filtered_point = self.point_history[-1]
            text += f"Fare 1: X: {last_filtered_point[0]:.2f}, Y: {last_filtered_point[1]:.2f}, Z: {last_filtered_point[2]:.2f}\n"
        else:
            text += "Tespit yok.\n"
        
        self.three_d_view_label.setText(text)


    def _convert_frame_to_pixmap(self, cv_img, label_widget):
        """OpenCV formatındaki görüntüyü Qt formatına dönüştürür ve boyutu sabit tutar."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled_img = convert_to_Qt_format.scaled(label_widget.width(), label_widget.height(), 
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(scaled_img)

    def update_slider_and_time(self):
        """Slider ve zaman etiketini günceller."""
        cap = next((vc for vc in self.video_captures if vc and vc.isOpened() and vc.get(cv2.CAP_PROP_FRAME_COUNT)>0), None)
        if cap and not self.progress_slider.isSliderDown():
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if total_frames > 0:
                self.progress_slider.setValue(int(current_frame * 100 / total_frames))
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                total_seconds = total_frames / fps
                current_seconds = current_frame / fps
            else:
                total_seconds = 0
                current_seconds = 0

            def format_time(s):
                return f"{int(s/60):02d}:{int(s%60):02d}"
            self.time_label.setText(f"{format_time(current_seconds)} / {format_time(total_seconds)}")
        elif not cap:
            self.time_label.setText("00:00 / 00:00")
            self.progress_slider.setValue(0)

    def closeEvent(self, event):
        """Uygulama kapatıldığında kaynakları serbest bırakır."""
        self.timer.stop()
        for vc in self.video_captures:
            if vc: vc.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    dark_stylesheet = """
        QWidget {
            background-color: #2E2E2E; color: #E0E0E0; font-size: 10pt;
        }
        QMainWindow, QDialog { background-color: #2E2E2E; }
        QLabel { border: 1px solid #555; }
        VideoLabel { border: 2px solid #C0C0C0; }
        MplCanvas { border: 2px solid #C0C0C0; }
        QPushButton {
            background-color: #555555; border: 1px solid #777777;
            padding: 5px; min-width: 70px; border-radius: 3px;
            color: #E0E0E0;
        }
        QPushButton:hover { background-color: #777777; border: 1px solid #999999; }
        QPushButton:pressed { background-color: #444444; }
        QPushButton:disabled { background-color: #404040; color: #888888; }
        QSlider::groove:horizontal {
            border: 1px solid #555555; height: 8px; background: #444444;
            margin: 2px 0; border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #00FF7F; border: 1px solid #00FF7F;
            width: 16px; margin: -4px 0; border-radius: 8px;
        }
        QMenuBar { background-color: #444444; }
        QMenuBar::item:selected { background-color: #777777; }
        QMenu { background-color: #444444; border: 1px solid #777777; }
        QMenu::item:selected { background-color: #777777; }
        QStatusBar { background-color: #444444; }
        QMessageBox { background-color: #2E2E2E; }
        QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #444444; border: 1px solid #777777;
            color: #E0E0E0; padding: 2px; border-radius: 3px;
        }
        QGroupBox {
            border: 1px solid #777777;
            margin-top: 10px;
            padding-top: 15px;
            font-weight: bold;
            color: #00FF7F;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center; /* center if at top, or top center */
            padding: 0 3px;
            background-color: #2E2E2E;
        }
    """
    app.setStyleSheet(dark_stylesheet)

    window = VideoPlayerWindow()
    window.show()
    sys.exit(app.exec_())