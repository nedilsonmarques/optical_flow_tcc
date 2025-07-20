from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
)
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
from processing.flow_calculators import FlowCalculator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Controle para Montagem Equatorial")
        
        # Configurações iniciais
        self.frame_delay = 100  # ms entre frames
        self.cap = cv2.VideoCapture(0)
        
        # Inicializar interface
        self.init_ui()
        self.init_camera()
        
        # Timer para atualização
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.frame_delay)
    
    def init_ui(self):
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        
        # Área de imagem
        self.image_label = QLabel()
        layout.addWidget(self.image_label, 60)
        
        # Controles
        control_panel = QVBoxLayout()
        
        # Slider para delay entre frames
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(10, 1000)
        self.delay_slider.setValue(self.frame_delay)
        self.delay_slider.valueChanged.connect(self.update_delay)
        control_panel.addWidget(QLabel("Delay entre frames (ms):"))
        control_panel.addWidget(self.delay_slider)
        
        layout.addLayout(control_panel, 40)
        self.setCentralWidget(central_widget)
    
    def init_camera(self):
        if not self.cap.isOpened():
            print("Erro ao abrir câmera")
            return
        
        # Testar primeiro frame
        ret, frame = self.cap.read()
        if not ret:
            print("Erro ao capturar frame")
            return
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Converter para exibição no Qt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(
                frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
    
    def update_delay(self, value):
        self.frame_delay = value
        self.timer.setInterval(value)
    
    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)