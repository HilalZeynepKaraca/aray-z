import sys
import cv2
import numpy as np
import os
import json
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QGroupBox, QMessageBox, QFileDialog, QSizePolicy, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

CALIBRATION_FILE = "calibration_data.json"

class CalibrationDialog(QDialog):
    # calibration_done_signal'a R_02 ve T_02 parametreleri eklendi
    calibration_done_signal = pyqtSignal(list, list, object, object, object, object)

    def __init__(self, parent=None, video_captures=None):
        super().__init__(parent)
        self.setWindowTitle("Kamera Kalibrasyonu")
        self.setFixedSize(1000, 700)

        self.video_captures = video_captures if video_captures else [None] * 3
        self.current_camera_index = 0

        self.chessboard_size = (9, 6)
        self.square_size = 20.0

        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size

        self.objpoints_list = [[] for _ in range(3)]
        self.imgpoints_list = [[] for _ in range(3)]
        
        self.camera_matrices = [None] * 3
        self.dist_coeffs = [None] * 3
        
        self.R_01 = None
        self.T_01 = None
        self.R_02 = None # Kamera 0-2 stereo için
        self.T_02 = None # Kamera 0-2 stereo için

        self._init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_camera_feed)
        self.timer.start(50)

        # Kalibrasyon diyalogu açıldığında kullanıcıya ipucu ver
        QMessageBox.information(self, "Kalibrasyon Bilgisi",
                                "Eğer fiziksel bir satranç tahtanız yoksa, 3D konumlandırmayı test etmek için 'Varsayılan Kalibrasyonu Kullan' butonuna tıklayabilirsiniz. Aksi takdirde, her kamera için en az 10 görüntü yakalayıp kalibrasyon yapmalısınız.")


    def _init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        info_layout = QHBoxLayout()
        self.camera_selector = QSpinBox()
        self.camera_selector.setRange(0, 2)
        self.camera_selector.setValue(self.current_camera_index)
        self.camera_selector.valueChanged.connect(self._change_camera)
        info_layout.addWidget(QLabel("Kalibre Edilen Kamera:"))
        info_layout.addWidget(self.camera_selector)
        
        self.captured_images_label = QLabel(f"Kamera 1: Yakalanan Görüntü: 0")
        info_layout.addWidget(self.captured_images_label)
        info_layout.addStretch()
        main_layout.addLayout(info_layout)

        self.video_display_label = QLabel("Kamera görüntüsü burada olacak")
        self.video_display_label.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_display_label)

        params_groupbox = QGroupBox("Satranç Tahtası Parametreleri")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("Genişlik (Köşe Sayısı):"), 0, 0)
        self.chess_w_edit = QSpinBox()
        self.chess_w_edit.setRange(1, 20)
        self.chess_w_edit.setValue(self.chessboard_size[0])
        self.chess_w_edit.valueChanged.connect(self._update_chessboard_size)
        params_layout.addWidget(self.chess_w_edit, 0, 1)

        params_layout.addWidget(QLabel("Yükseklik (Köşe Sayısı):"), 1, 0)
        self.chess_h_edit = QSpinBox()
        self.chess_h_edit.setRange(1, 20)
        self.chess_h_edit.setValue(self.chessboard_size[1])
        self.chess_h_edit.valueChanged.connect(self._update_chessboard_size)
        params_layout.addWidget(self.chess_h_edit, 1, 1)

        params_layout.addWidget(QLabel("Kare Boyutu (mm):"), 2, 0)
        self.square_size_edit = QLineEdit(str(self.square_size))
        self.square_size_edit.textChanged.connect(self._update_square_size)
        params_layout.addWidget(self.square_size_edit, 2, 1)

        params_groupbox.setLayout(params_layout)
        main_layout.addWidget(params_groupbox)

        buttons_layout = QHBoxLayout()
        self.capture_button = QPushButton("Görüntü Yakala")
        self.capture_button.clicked.connect(self._capture_image)
        buttons_layout.addWidget(self.capture_button)

        self.calibrate_button = QPushButton("Kamera Kalibre Et")
        self.calibrate_button.clicked.connect(self._perform_calibration)
        self.calibrate_button.setEnabled(False)
        buttons_layout.addWidget(self.calibrate_button)

        self.stereo_01_button = QPushButton("Stereo Kalibre Et (Kamera 0-1)")
        self.stereo_01_button.clicked.connect(self._perform_stereo_calibration_01)
        self.stereo_01_button.setEnabled(False)
        buttons_layout.addWidget(self.stereo_01_button)
        
        self.stereo_02_button = QPushButton("Stereo Kalibre Et (Kamera 0-2)")
        self.stereo_02_button.clicked.connect(self._perform_stereo_calibration_02)
        self.stereo_02_button.setEnabled(False)
        buttons_layout.addWidget(self.stereo_02_button)
        
        self.save_button = QPushButton("Tüm Kalibrasyonları Kaydet")
        self.save_button.clicked.connect(self._save_calibration_data)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)

        # Yeni buton: Varsayılan Kalibrasyonu Kullan
        self.use_default_calib_button = QPushButton("Varsayılan Kalibrasyonu Kullan")
        self.use_default_calib_button.clicked.connect(self._use_default_calibration)
        buttons_layout.addWidget(self.use_default_calib_button)


        self.ok_button = QPushButton("Tamam")
        self.ok_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.ok_button)

        main_layout.addLayout(buttons_layout)
        self._update_status_labels()

    def _change_camera(self, index):
        self.current_camera_index = index
        self._update_status_labels()

    def _update_chessboard_size(self):
        self.chessboard_size = (self.chess_w_edit.value(), self.chess_h_edit.value())
        self._update_objp()
        self._update_status_labels()

    def _update_square_size(self):
        try:
            self.square_size = float(self.square_size_edit.text())
            self._update_objp()
        except ValueError:
            pass
        self._update_status_labels()

    def _update_objp(self):
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size

    def _update_status_labels(self):
        self.captured_images_label.setText(
            f"Kamera {self.current_camera_index+1}: Yakalanan Görüntü: {len(self.imgpoints_list[self.current_camera_index])}")
        
        self.calibrate_button.setEnabled(len(self.imgpoints_list[self.current_camera_index]) >= 10)
        self.stereo_01_button.setEnabled(
            len(self.imgpoints_list[0]) >= 10 and 
            len(self.imgpoints_list[1]) >= 10 and 
            self.camera_matrices[0] is not None and 
            self.camera_matrices[1] is not None
        )
        self.stereo_02_button.setEnabled(
            len(self.imgpoints_list[0]) >= 10 and 
            len(self.imgpoints_list[2]) >= 10 and 
            self.camera_matrices[0] is not None and 
            self.camera_matrices[2] is not None
        )
        self.save_button.setEnabled(
            any(m is not None for m in self.camera_matrices) or
            self.R_01 is not None or
            self.R_02 is not None
        )

    def _update_camera_feed(self):
        if self.video_captures[self.current_camera_index] and self.video_captures[self.current_camera_index].isOpened():
            ret, frame = self.video_captures[self.current_camera_index].read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

                display_frame = frame.copy()
                if ret_corners:
                    cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret_corners)
                    self.video_display_label.setText(f"Kamera {self.current_camera_index+1} (Köşeler Tespit Edildi)")
                else:
                    self.video_display_label.setText(f"Kamera {self.current_camera_index+1} (Köşeler Tespit Edilemedi)")
                
                h, w, ch = display_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)
                
                label_rect = self.video_display_label.contentsRect()
                scaled_pixmap = pixmap.scaled(label_rect.width(), label_rect.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_display_label.setPixmap(scaled_pixmap)
            else:
                self.video_display_label.setText(f"Kamera {self.current_camera_index+1} kare okunamadı.")
        else:
            self.video_display_label.setText(f"Kamera {self.current_camera_index+1} mevcut değil veya açılamadı.")
            
    def _capture_image(self):
        if self.video_captures[self.current_camera_index] and self.video_captures[self.current_camera_index].isOpened():
            ret, frame = self.video_captures[self.current_camera_index].read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

                if ret_corners:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    
                    self.imgpoints_list[self.current_camera_index].append(corners)
                    self.objpoints_list[self.current_camera_index].append(self.objp)
                    self._update_status_labels()
                    QMessageBox.information(self, "Görüntü Yakalandı", 
                                             f"Kamera {self.current_camera_index+1} için {len(self.imgpoints_list[self.current_camera_index])}. görüntü yakalandı.")
                else:
                    QMessageBox.warning(self, "Uyarı", "Satranç tahtası köşeleri tespit edilemedi. Lütfen tahtayı daha net gösterin.")
            else:
                QMessageBox.critical(self, "Hata", "Kamera karesi okunamadı.")
        else:
            QMessageBox.critical(self, "Hata", "Kamera mevcut değil veya açılamadı.")

    def _perform_calibration(self):
        cam_idx = self.current_camera_index
        if len(self.imgpoints_list[cam_idx]) < 10:
            QMessageBox.warning(self, "Uyarı", f"Kalibrasyon için en az 10 görüntüye ihtiyacınız var. Şu anda {len(self.imgpoints_list[cam_idx])} görüntü yakalandı.")
            return

        ret, temp_frame = self.video_captures[cam_idx].read()
        if not ret:
            QMessageBox.critical(self, "Hata", "Kalibrasyon için görüntü boyutu alınamadı.")
            return
        
        h, w = temp_frame.shape[:2]
        img_size = (w, h)

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints_list[cam_idx],
                self.imgpoints_list[cam_idx],
                img_size, None, None
            )
            if ret:
                self.camera_matrices[cam_idx] = mtx
                self.dist_coeffs[cam_idx] = dist
                QMessageBox.information(self, "Kalibrasyon Başarılı", 
                                         f"Kamera {cam_idx+1} kalibrasyonu başarıyla tamamlandı. Yeniden projeksiyon hatası: {ret:.4f}")
            else:
                QMessageBox.critical(self, "Kalibrasyon Hatası", f"Kamera {cam_idx+1} kalibrasyonu başarısız oldu.")
        except Exception as e:
            QMessageBox.critical(self, "Kalibrasyon Hatası", f"Kalibrasyon sırasında bir hata oluştu: {e}")
        finally:
            self._update_status_labels()

    def _perform_stereo_calibration_01(self):
        # Stereo kalibrasyon için Kamera 0 ve Kamera 1'in verilerini kullan
        if not (self.camera_matrices[0] is not None and self.camera_matrices[1] is not None):
            QMessageBox.warning(self, "Uyarı", "Stereo kalibrasyon için önce Kamera 0 ve Kamera 1'in tekli kalibrasyonunu yapmalısınız.")
            return
        
        if len(self.imgpoints_list[0]) < 10 or len(self.imgpoints_list[1]) < 10:
            QMessageBox.warning(self, "Uyarı", "Stereo kalibrasyon için her iki kamera için de en az 10 görüntüye ihtiyacınız var.")
            return

        ret, temp_frame = self.video_captures[0].read()
        if not ret:
            QMessageBox.critical(self, "Hata", "Stereo kalibrasyon için görüntü boyutu alınamadı.")
            return
        img_size = temp_frame.shape[:2][::-1]

        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
                self.objpoints_list[0], self.imgpoints_list[0], self.imgpoints_list[1],
                self.camera_matrices[0], self.dist_coeffs[0],
                self.camera_matrices[1], self.dist_coeffs[1],
                img_size,
                criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
            )

            if ret:
                self.R_01 = R
                self.T_01 = T
                QMessageBox.information(self, "Stereo Kalibrasyon Başarılı", 
                                        f"Kamera 0-1 stereo kalibrasyon başarıyla tamamlandı. Yeniden projeksiyon hatası: {ret:.4f}")
            else:
                QMessageBox.critical(self, "Stereo Kalibrasyon Hatası", "Kamera 0-1 stereo kalibrasyonu başarısız oldu.")
        except Exception as e:
            QMessageBox.critical(self, "Stereo Kalibrasyon Hatası", f"Kamera 0-1 kalibrasyonu sırasında bir hata oluştu: {e}")
        finally:
            self._update_status_labels()

    def _perform_stereo_calibration_02(self):
        # Stereo kalibrasyon için Kamera 0 ve Kamera 2'nin verilerini kullan
        if not (self.camera_matrices[0] is not None and self.camera_matrices[2] is not None):
            QMessageBox.warning(self, "Uyarı", "Stereo kalibrasyon için önce Kamera 0 ve Kamera 2'nin tekli kalibrasyonunu yapmalısınız.")
            return
        
        if len(self.imgpoints_list[0]) < 10 or len(self.imgpoints_list[2]) < 10:
            QMessageBox.warning(self, "Uyarı", "Stereo kalibrasyon için her iki kamera için de en az 10 görüntüye ihtiyacınız var.")
            return

        ret, temp_frame = self.video_captures[0].read()
        if not ret:
            QMessageBox.critical(self, "Hata", "Stereo kalibrasyon için görüntü boyutu alınamadı.")
            return
        img_size = temp_frame.shape[:2][::-1]

        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
                self.objpoints_list[0], self.imgpoints_list[0], self.imgpoints_list[2],
                self.camera_matrices[0], self.dist_coeffs[0],
                self.camera_matrices[2], self.dist_coeffs[2],
                img_size,
                criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
            )

            if ret:
                self.R_02 = R
                self.T_02 = T
                QMessageBox.information(self, "Stereo Kalibrasyon Başarılı", 
                                        f"Kamera 0-2 stereo kalibrasyon başarıyla tamamlandı. Yeniden projeksiyon hatası: {ret:.4f}")
            else:
                QMessageBox.critical(self, "Stereo Kalibrasyon Hatası", "Kamera 0-2 stereo kalibrasyonu başarısız oldu.")
        except Exception as e:
            QMessageBox.critical(self, "Stereo Kalibrasyon Hatası", f"Kamera 0-2 kalibrasyonu sırasında bir hata oluştu: {e}")
        finally:
            self._update_status_labels()

    def _use_default_calibration(self):
        """Satranç tahtası olmadan varsayılan kalibrasyon verilerini ayarlar."""
        W, H = 1280, 720
        default_K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float32)
        default_D = np.zeros(5, dtype=np.float32)
        for i in range(3):
            self.camera_matrices[i] = default_K
            self.dist_coeffs[i] = default_D
        
        self.R_01 = np.eye(3, dtype=np.float32)
        self.T_01 = np.array([[30.0], [0.0], [0.0]], dtype=np.float32) # X ekseninde 30 birim öteleme
        
        angle = np.pi / 2 # 90 derece rotasyon
        self.R_02 = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        self.T_02 = np.array([[0.0], [30.0], [0.0]], dtype=np.float32) # Y ekseninde 30 birim öteleme
        
        self.is_calibrated = True
        QMessageBox.information(self, "Varsayılan Kalibrasyon", "Varsayılan kalibrasyon verileri yüklendi.")
        self.calibration_done_signal.emit(self.camera_matrices, self.dist_coeffs, self.R_01, self.T_01, self.R_02, self.T_02)
        self.accept()

    def _save_calibration_data(self):
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
            self.calibration_done_signal.emit(self.camera_matrices, self.dist_coeffs, self.R_01, self.T_01, self.R_02, self.T_02)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kalibrasyon verileri kaydedilirken bir hata oluştu: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)