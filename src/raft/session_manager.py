# src/raft/session_manager.py

import os
import json
from datetime import datetime
import cv2

class SessionManager:
    def __init__(self, base_folder):
        self.session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_name = f"session_{self.session_timestamp}"
        self.session_dir = os.path.join(base_folder, self.session_name)
        self.images_dir = os.path.join(self.session_dir, "images")
        self.metadata_path = os.path.join(self.session_dir, "metadata.json")
        self.data_csv_path = os.path.join(self.session_dir, "data.csv")
        self.frame_count = 0
        self.outliers_rejected = 0

    def _write_metadata(self, metadata_dict):
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=4)

    def start_session(self, initial_metadata):
        print("-" * 50)
        print(f"Iniciando nova sessão: {self.session_name}")
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"Diretório da sessão criado em: {self.session_dir}")

        initial_metadata['session_id'] = self.session_timestamp
        self._write_metadata(initial_metadata)
        print(f"Metadados iniciais salvos em: {self.metadata_path}")

        # MODIFICADO: Cabeçalho correto e final
        with open(self.data_csv_path, 'w', encoding='utf-8') as f:
            f.write("elapsed_time_s,speed_px_s,flow_x_px,flow_y_px\n")
        print(f"Arquivo de dados criado em: {self.data_csv_path}")
        print("-" * 50)

    def log_data_row(self, elapsed_time, speed_px_s, flow_x, flow_y):
        """MODIFICADO: Assinatura da função e escrita da linha corrigidas."""
        with open(self.data_csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{elapsed_time:.4f},{speed_px_s:.4f},{flow_x:.4f},{flow_y:.4f}\n")

    def save_image_frame(self, frame):
        self.frame_count += 1
        filename = f"frame_{self.frame_count:06d}.png"
        filepath = os.path.join(self.images_dir, filename)
        cv2.imwrite(filepath, frame)

    def end_session(self, final_metadata_update):
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata.update(final_metadata_update)
            metadata['total_frames_saved'] = self.frame_count
            metadata['total_outliers_rejected'] = self.outliers_rejected
            self._write_metadata(metadata)
            print(f"Metadados finais salvos em: {self.metadata_path}")
        except Exception as e:
            print(f"Erro ao finalizar a sessão e salvar metadados: {e}")