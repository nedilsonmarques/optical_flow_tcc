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

from . import config, analysis, visualization

def get_images_from_folder(folder_path):
    """Calcula o FPS de captura e retorna um gerador de imagens e timestamps."""
    try:
        files_with_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        files_with_paths = [f for f in files_with_paths if os.path.isfile(f)]
        files_with_paths.sort(key=os.path.getmtime)
    except FileNotFoundError:
        print(f"ERRO: A pasta de imagens '{folder_path}' não foi encontrada.")
        return None, 0

    image_files = [f for f in files_with_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print(f"ERRO: Nenhuma imagem encontrada na pasta '{folder_path}'.")
        return None, 0

    timestamps_ms = [os.path.getmtime(f) * 1000 for f in image_files]
    deltas_ms = np.diff(timestamps_ms)
    median_delta_s = np.median(deltas_ms) / 1000.0 if len(deltas_ms) > 0 else 0
    capture_fps = 1.0 / median_delta_s if median_delta_s > 0 else 0

    print(f"Encontradas {len(image_files)} imagens.")
    
    def image_generator():
        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is not None:
                yield frame, timestamps_ms[i]
            else:
                print(f"Aviso: Não foi possível ler a imagem {image_path}")
    
    return image_generator(), capture_fps

def main():
    print("Configurando o dispositivo e o modelo RAFT...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Processando no dispositivo: {device.upper()} ---")

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device)
    model.eval(); transforms = weights.transforms()
    
    fig, axes, lines, annotations, harmonic_lines = visualization.create_plots()

    time_data, speed_data_raw_px_s, position_total_px, calibration_factor_K = [], [], 0.0, 0.0
    last_plot_update = time.time()
    
    # --- Lógica de Inicialização ---
    if config.USE_LIVE_CAMERA:
        print("--- MODO: Câmera Ao Vivo ---")
        data_source = cv2.VideoCapture(0)
        if not data_source.isOpened(): print("Erro: Câmera não encontrada."); return
        print("Câmera inicializada com sucesso.")
        
        print("Medindo FPS da câmera ao vivo por 5 segundos...")
        fps_measurements = []
        start_fps_measure = time.time()
        # Captura alguns frames para medir o FPS real da câmera
        for _ in range(150):
             if time.time() - start_fps_measure > 5.0: break
             t0 = time.time(); ret, _ = data_source.read(); t1 = time.time()
             if not ret: continue
             fps_sample = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
             if fps_sample > 0: fps_measurements.append(fps_sample)
        avg_fps = np.mean(fps_measurements) if fps_measurements else 30.0
    else:
        print(f"--- MODO: Análise Offline de Arquivos ---")
        print(f"Analisando timestamps em: {config.IMAGE_FOLDER_PATH}")
        data_source, avg_fps = get_images_from_folder(config.IMAGE_FOLDER_PATH)
        if data_source is None: return

    maf_short_points = max(1, int(config.MOVING_AVERAGE_SHORT_SECONDS * avg_fps))
    maf_long_points = max(1, int(config.MOVING_AVERAGE_LONG_SECONDS * avg_fps))
    print("-" * 50)
    print(f"FPS base para filtros: {avg_fps:.2f} ({(1/avg_fps)*1000 if avg_fps>0 else 0:.1f} ms/quadro)")
    print(f"Filtro de Média Móvel Curto definido para {maf_short_points} pontos.")
    print(f"Filtro de Média Móvel Longo definido para {maf_long_points} pontos.")
    print("-" * 50)

    # Obtenção do primeiro quadro
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
    processing_start_time = datetime.now()

    # --- Loop de Processamento Principal ---
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
        delta_time_s = (current_msec - prev_msec) / 1000.0 if current_msec != prev_msec else 1e-6

        prev_frame_tensor = F.to_tensor(prev_frame_np).unsqueeze(0)
        current_frame_tensor = F.to_tensor(current_frame_np).unsqueeze(0)
        img1_batch, img2_batch = transforms(prev_frame_tensor, current_frame_tensor)
        img1_batch = img1_batch.to(device); img2_batch = img2_batch.to(device)
        flow_start_time = time.time()
        with torch.no_grad(): list_of_flows = model(img1_batch, img2_batch)
        flow_proc_time = time.time() - flow_start_time
        predicted_flow = list_of_flows[-1]
        magnitude = torch.sqrt(predicted_flow[0, 0]**2 + predicted_flow[0, 1]**2)
        avg_magnitude_px = magnitude.mean().item()
        current_speed_px_s = avg_magnitude_px / delta_time_s
        
        is_outlier = False
        if len(speed_data_raw_px_s) > maf_short_points:
            mean = np.mean(speed_data_raw_px_s); std = np.std(speed_data_raw_px_s)
            upper = mean + config.OUTLIER_STD_DEV_THRESHOLD * std
            lower = max(0, mean - config.OUTLIER_STD_DEV_THRESHOLD * std)
            if not (lower <= current_speed_px_s <= upper): is_outlier = True
        
        if not is_outlier:
            elapsed_time = (current_msec - start_msec) / 1000.0
            time_data.append(elapsed_time)
            speed_data_raw_px_s.append(current_speed_px_s)
            position_total_px += current_speed_px_s * delta_time_s
            
            if len(speed_data_raw_px_s) >= maf_long_points:
                latest_avg_px_s = np.mean(speed_data_raw_px_s[-maf_long_points:])
                if latest_avg_px_s > 0:
                    calibration_factor_K = config.THEORETICAL_SPEED_ARCSEC_S / latest_avg_px_s
        
        if time.time() - last_plot_update > config.PLOT_UPDATE_SECONDS:
            analysis_results = analysis.process_analysis_data(time_data, speed_data_raw_px_s, calibration_factor_K, maf_short_points, maf_long_points)
            visualization.update_plots(fig, axes, lines, annotations, harmonic_lines, analysis_results)
            last_plot_update = time.time()
        
        vis_frame = visualization.draw_flow(current_frame_np, predicted_flow[0])
        metrics = {"fps": processing_fps, "flow_time_ms": flow_proc_time * 1000, "delta_time_ms": delta_time_s * 1000,
                   "speed_norm_arcsec_s": (current_speed_px_s * calibration_factor_K) / config.SPEED_FACTOR,
                   "pos_total_arcsec": position_total_px * calibration_factor_K}
        vis_frame = visualization.draw_hud(vis_frame, metrics)
        cv2.imshow('Fluxo Optico', vis_frame)
        
        plt.pause(1e-3)
        if cv2.getWindowProperty('Fluxo Optico', cv2.WND_PROP_VISIBLE) < 1: break
        
        prev_frame_np = current_frame_np.copy()
        prev_msec = current_msec
    
    # --- Loop de Espera e Resumo Final (Apenas para modo offline) ---
    if not config.USE_LIVE_CAMERA:
        processing_end_time = datetime.now()
        duration = processing_end_time - processing_start_time
        print("\n" + "="*50)
        print("Processamento de arquivos concluído.")
        print(f"  - Horário de Início: {processing_start_time.strftime('%H:%M:%S')}")
        print(f"  - Horário de Fim:    {processing_end_time.strftime('%H:%M:%S')}")
        print(f"  - Duração Total:     {str(duration).split('.')[0]}")
        print("="*50)
        print("Mantendo os gráficos abertos para análise. Feche a janela do gráfico para encerrar.")

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
            if cv2.getWindowProperty('Fluxo Optico', cv2.WND_PROP_VISIBLE) < 1: break
    
    # --- Limpeza ---
    if config.USE_LIVE_CAMERA and 'data_source' in locals() and data_source.isOpened():
        data_source.release(); print("Câmera liberada.")
    plt.ioff(); plt.close('all'); print("Gráficos fechados.")
    cv2.destroyAllWindows(); print("Janelas fechadas. Programa encerrado.")

# NOVO BLOCO PARA SUBSTITUIR O ANTIGO
if __name__ == '__main__':
    import traceback # Importa o módulo de traceback
    try:
        main()
    except Exception as e:
        # Agora imprimimos um relatório de erro muito mais detalhado
        print("\n" + "#"*60)
        print("### OCORREU UM ERRO FATAL ###")
        print(f"Tipo de Erro: {type(e).__name__}")
        print(f"Mensagem: {e}")
        print("\n--- RASTREAMENTO COMPLETO (TRACEBACK) ---")
        traceback.print_exc() # Imprime a pilha de chamadas completa
        print("#"*60 + "\n")
    finally:
        plt.ioff(); plt.close('all'); print("Gráficos fechados.")
        cv2.destroyAllWindows(); print("Janelas fechadas. Programa encerrado.")