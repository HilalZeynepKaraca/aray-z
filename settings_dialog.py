import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox, QMessageBox, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal

class SettingsDialog(QDialog):
    # conf_thresholds artık bir liste olacak
    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Giriş ve Model Ayarları")
        self.setGeometry(200, 200, 600, 400)

        self.video_input_paths = ["", "", ""]
        self.onnx_model_paths = ["", "", ""]
        self.conf_thresholds = [0.1, 0.1, 0.1] # Her kamera için varsayılan eşik
        self.iou_threshold = 0.45

        if initial_settings:
            self.video_input_paths = initial_settings.get("input_paths", ["", "", ""])
            self.onnx_model_paths = initial_settings.get("onnx_paths", ["", "", ""])
            # Eğer initial_settings'te conf_thresholds yoksa veya tek bir değerse, bunu listeye dönüştür
            if "conf_thresholds" in initial_settings and isinstance(initial_settings["conf_thresholds"], list):
                self.conf_thresholds = initial_settings["conf_thresholds"]
            elif "conf_threshold" in initial_settings: # Eski tekil conf_threshold'u destekle
                val = initial_settings["conf_threshold"]
                self.conf_thresholds = [val, val, val]
            
            self.iou_threshold = initial_settings.get("iou_threshold", 0.45)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()

        # Video Girişleri ve ONNX Model Yolları
        for i in range(3):
            video_group_box = QHBoxLayout()
            video_label = QLabel(f"Video {i+1} Kaynağı:")
            self.video_path_edits = getattr(self, 'video_path_edits', [])
            self.video_path_edits.append(QLineEdit(self.video_input_paths[i]))
            self.video_path_edits[i].setPlaceholderText(f"Video {i+1} dosya yolu veya kamera indeksi (0, 1, ...)")
            browse_video_button = QPushButton("Gözat...")
            browse_video_button.clicked.connect(lambda _, idx=i: self._browse_file(self.video_path_edits[idx]))

            video_group_box.addWidget(video_label)
            video_group_box.addWidget(self.video_path_edits[i])
            video_group_box.addWidget(browse_video_button)
            main_layout.addLayout(video_group_box)

            onnx_group_box = QHBoxLayout()
            onnx_label = QLabel(f"ONNX Model {i+1} Yolu:")
            self.onnx_path_edits = getattr(self, 'onnx_path_edits', [])
            self.onnx_path_edits.append(QLineEdit(self.onnx_model_paths[i]))
            self.onnx_path_edits[i].setPlaceholderText(f"ONNX Model {i+1} dosya yolu (.onnx)")
            browse_onnx_button = QPushButton("Gözat...")
            browse_onnx_button.clicked.connect(lambda _, idx=i: self._browse_file(self.onnx_path_edits[idx], filter="ONNX Modelleri (*.onnx)"))

            onnx_group_box.addWidget(onnx_label)
            onnx_group_box.addWidget(self.onnx_path_edits[i])
            onnx_group_box.addWidget(browse_onnx_button)
            main_layout.addLayout(onnx_group_box)
            
            # Her kamera için ayrı CONF_THRESHOLD
            conf_threshold_group_box = QHBoxLayout()
            conf_threshold_label = QLabel(f"Kamera {i+1} Güven Eşiği (CONF_THRESHOLD):")
            self.conf_threshold_spinboxes = getattr(self, 'conf_threshold_spinboxes', [])
            spin_box = QDoubleSpinBox()
            spin_box.setRange(0.001, 1.0)
            spin_box.setSingleStep(0.01)
            spin_box.setDecimals(3)
            spin_box.setValue(self.conf_thresholds[i])
            self.conf_threshold_spinboxes.append(spin_box)

            conf_threshold_group_box.addWidget(conf_threshold_label)
            conf_threshold_group_box.addWidget(self.conf_threshold_spinboxes[i])
            main_layout.addLayout(conf_threshold_group_box)


        # IOU Threshold
        iou_layout = QHBoxLayout()
        iou_label = QLabel("NMS IOU Eşiği (IOU_THRESHOLD):")
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.0)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setDecimals(2)
        self.iou_spinbox.setValue(self.iou_threshold)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_spinbox)
        main_layout.addLayout(iou_layout)

        # Butonlar
        button_layout = QHBoxLayout()
        save_button = QPushButton("Kaydet")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("İptal")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _browse_file(self, line_edit, filter="Tüm Dosyalar (*.*)"):
        file_name, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "", filter)
        if file_name:
            line_edit.setText(file_name)

    def get_settings(self):
        settings = {
            "input_paths": [edit.text() for edit in self.video_path_edits],
            "onnx_paths": [edit.text() for edit in self.onnx_path_edits],
            "conf_thresholds": [spinbox.value() for spinbox in self.conf_threshold_spinboxes], # Liste olarak alın
            "iou_threshold": self.iou_spinbox.value()
        }
        return settings

