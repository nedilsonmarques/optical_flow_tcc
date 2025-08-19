# src/raft/main_live.py

import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback

from . import config, analysis, visualization, session_manager

def get_images_from_folder(folder_path):
    """
    MODIFICADO: Calcula o FPS de captura usando o tempo total / número de frames.
    """
    try:
        files_with_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        files_with_paths = [f for f in files_with_paths if os.path.isfile(f)]
        files_with_paths.sort(key=os.path.getctime)
    except FileNotFoundError:
        print(f"ERRO: A pasta de imagens '{folder_path}' não foi encontrada.")
        return None, 0

    image_files = [f for f in files_with_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files or len(image_files) < 2: # Precisa de pelo menos 2 imagens para ter uma duração
        print(f"ERRO: Número insuficiente de imagens encontrado na pasta '{folder_path}'.")
        return None, 0

    timestamps_ms = [os.path.getctime(f) * 1000 for f in image_files]
    
    # --- MODIFICADO: Cálculo de FPS simplificado e mais robusto ---
    print('Inicio: ', timestamps_ms[-1])
    print('Fim: ', timestamps_ms[0])
    total_duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    num_frames = len(image_files)
    capture_fps = (num_frames - 1) / total_duration_s if total_duration_s > 0 else 0

    print(f"Encontradas {num_frames} imagens.")
    
    def image_generator():
        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is not None:
                yield frame, timestamps_ms[i]
            else:
                print(f"Aviso: Não foi possível ler a imagem {image_path}")
    
    return image_generator(), capture_fps

def print_filter_settings(avg_fps, maf_short_points, maf_long_points):
    """ Imprime as configurações dos filtros dinâmicos. """
    print("-" * 50)
    print(f"FPS base para filtros: {avg_fps:.2f} ({(1/avg_fps)*1000 if avg_fps>0 else 0:.1f} ms/quadro)")
    print(f"Filtro de Média Móvel Curto definido para {maf_short_points} pontos.")
    print(f"Filtro de Média Móvel Longo definido para {maf_long_points} pontos.")
    print("-" * 50)

def main():
    session = None
    try:
        print("Configurando o dispositivo e o modelo RAFT...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Processando no dispositivo: {device.upper()} ---")

        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).to(device)
        model.eval(); transforms = weights.transforms()
        
        session = session_manager.SessionManager(config.SESSIONS_BASE_FOLDER)
        initial_metadata = {
            "start_time_utc": datetime.utcnow().isoformat(),
            "mode": "Live Camera" if config.USE_LIVE_CAMERA else "Offline Files",
            "image_folder": config.IMAGE_FOLDER_PATH if not config.USE_LIVE_CAMERA else "N/A",
            "worm_gear_teeth": config.WORM_GEAR_TEETH, "speed_factor": config.SPEED_FACTOR
        }

        fig, axes, lines, annotations, harmonic_lines = visualization.create_plots()

        time_data, speed_data_raw_px_s, position_total_px, calibration_factor_K = [], [], 0.0, 0.0
        last_plot_update = time.time()
        maf_short_points, maf_long_points = None, None
        
        if config.USE_LIVE_CAMERA:
            print("--- MODO: Câmera Ao Vivo ---")
            data_source = cv2.VideoCapture(0)
            if not data_source.isOpened(): print("Erro: Câmera não encontrada."); return
            print("Câmera inicializada com sucesso.")
            initial_fps_measurements = []
            avg_fps = None
        else:
            print(f"--- MODO: Análise Offline de Arquivos ---")
            data_source, avg_fps = get_images_from_folder(config.IMAGE_FOLDER_PATH)
            if data_source is None: return

        if avg_fps is not None:
            maf_short_points = max(1, int(config.MOVING_AVERAGE_SHORT_SECONDS * avg_fps))
            maf_long_points = max(1, int(config.MOVING_AVERAGE_LONG_SECONDS * avg_fps))
            initial_metadata.update({'capture_fps': avg_fps, 'maf_short_points': maf_short_points, 'maf_long_points': maf_long_points})
            print_filter_settings(avg_fps, maf_short_points, maf_long_points)

        session.start_session(initial_metadata)

        if config.USE_LIVE_CAMERA:
            ret, prev_frame_np = data_source.read()
            if not ret: print("Erro: Não foi possível obter o primeiro quadro."); return
            start_msec = data_source.get(cv2.CAP_PROP_POS_MSEC); prev_msec = start_msec
        else:
            prev_frame_np, prev_msec = next(data_source, (None, None))
            if prev_frame_np is None: print("Erro: Não foi possível obter o primeiro quadro da pasta."); return
            start_msec = prev_msec

        print("\nIniciando o loop de processamento. Feche a janela do gráfico para sair.")
        prev_time_fps = time.time()

        while plt.fignum_exists(fig.number):
            if config.USE_LIVE_CAMERA:
                ret, current_frame_np = data_source.read()
                if not ret: break
                current_msec = data_source.get(cv2.CAP_PROP_POS_MSEC)
            else:
                current_frame_np, current_msec = next(data_source, (None, None))
                if current_frame_np is None: break

            current_time_fps = time.time()
            processing_fps = 1.0 / (current_time_fps - prev_time_fps) if current_time_fps != prev_time_fps else 0
            prev_time_fps = current_time_fps
            
            if config.USE_LIVE_CAMERA and maf_long_points is None:
                if len(initial_fps_measurements) < 50:
                    if processing_fps > 0: initial_fps_measurements.append(processing_fps)
                else:
                    avg_fps = np.mean(initial_fps_measurements)
                    maf_short_points = max(1, int(config.MOVING_AVERAGE_SHORT_SECONDS * avg_fps))
                    maf_long_points = max(1, int(config.MOVING_AVERAGE_LONG_SECONDS * avg_fps))
                    session.end_session({"processing_fps_measured": avg_fps, "maf_short_points": maf_short_points, "maf_long_points": maf_long_points})
                    print_filter_settings(avg_fps, maf_short_points, maf_long_points)
            
            delta_time_s = (current_msec - prev_msec) / 1000.0 if current_msec != prev_msec else 1e-6

            prev_frame_tensor = F.to_tensor(prev_frame_np).unsqueeze(0)
            current_frame_tensor = F.to_tensor(current_frame_np).unsqueeze(0)
            img1_batch, img2_batch = transforms(prev_frame_tensor, current_frame_tensor)
            img1_batch = img1_batch.to(device); img2_batch = img2_batch.to(device)
            flow_start_time = time.time()
            with torch.no_grad(): list_of_flows = model(img1_batch, img2_batch)
            flow_proc_time = time.time() - flow_start_time
            predicted_flow = list_of_flows[-1]
            
            avg_flow_x_px = predicted_flow[0, 0].mean().item()
            avg_flow_y_px = predicted_flow[0, 1].mean().item()
            avg_magnitude_px = np.sqrt(avg_flow_x_px**2 + avg_flow_y_px**2)
            current_speed_px_s = avg_magnitude_px / delta_time_s
            
            vis_frame = visualization.draw_flow(current_frame_np, predicted_flow[0])
            metrics = {"fps": processing_fps, "flow_time_ms": flow_proc_time * 1000, "delta_time_ms": delta_time_s * 1000,
                       "speed_norm_arcsec_s": (current_speed_px_s * calibration_factor_K) / config.SPEED_FACTOR,
                       "pos_total_arcsec": position_total_px * calibration_factor_K}
            
            if maf_long_points is not None:
                is_outlier = False
                if len(speed_data_raw_px_s) > maf_short_points:
                    mean = np.mean(speed_data_raw_px_s); std = np.std(speed_data_raw_px_s)
                    upper = mean + config.OUTLIER_STD_DEV_THRESHOLD * std
                    lower = max(0, mean - config.OUTLIER_STD_DEV_THRESHOLD * std)
                    if not (lower <= current_speed_px_s <= upper):
                        is_outlier = True
                
                if not is_outlier:
                    elapsed_time = (current_msec - start_msec) / 1000.0
                    time_data.append(elapsed_time)
                    speed_data_raw_px_s.append(current_speed_px_s)
                    position_total_px += current_speed_px_s * delta_time_s
                    
                    session.log_data_row(current_msec, avg_flow_x_px, avg_flow_y_px, avg_magnitude_px)
                    if config.USE_LIVE_CAMERA:
                        session.save_image_frame(current_frame_np)

                    if len(speed_data_raw_px_s) >= maf_long_points:
                        latest_avg_px_s = np.mean(speed_data_raw_px_s[-maf_long_points:])
                        if latest_avg_px_s > 0:
                            calibration_factor_K = config.THEORETICAL_SPEED_ARCSEC_S / latest_avg_px_s
                else:
                    session.outliers_rejected += 1

                if time.time() - last_plot_update > config.PLOT_UPDATE_SECONDS:
                    analysis_results = analysis.process_analysis_data(time_data, speed_data_raw_px_s, calibration_factor_K, maf_short_points, maf_long_points)
                    visualization.update_plots(fig, axes, lines, annotations, harmonic_lines, analysis_results)
                    last_plot_update = time.time()
                
                vis_frame = visualization.draw_hud(vis_frame, metrics)
            else:
                 cv2.putText(vis_frame, "Calibrando FPS...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            cv2.imshow('Fluxo Optico', vis_frame)
            
            plt.pause(1e-3)
            if cv2.getWindowProperty('Fluxo Optico', cv2.WND_PROP_VISIBLE) < 1: break
            
            prev_frame_np = current_frame_np.copy()
            prev_msec = current_msec
        
        final_metadata = {
            "end_time_utc": datetime.utcnow().isoformat(),
            "total_valid_frames_processed": len(time_data)
        }
        session.end_session(final_metadata)

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        traceback.print_exc()
    finally:
        if 'data_source' in locals() and config.USE_LIVE_CAMERA and data_source.isOpened():
            data_source.release(); print("Câmera liberada.")
        plt.ioff(); plt.close('all'); print("Gráficos fechados.")
        cv2.destroyAllWindows(); print("Janelas fechadas. Programa encerrado.")

if __name__ == '__main__':
    import traceback
    main()