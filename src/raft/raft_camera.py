import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import detrend, find_peaks # ADICIONADO: find_peaks

# --- Configurações ---
PLOT_UPDATE_SECONDS = 1.0
MOVING_AVERAGE_WINDOW_SHORT = 30
MOVING_AVERAGE_WINDOW_LONG = 300
OUTLIER_STD_DEV_THRESHOLD = 3.0 

# NOVO: Parâmetros para a detecção de picos na FFT
FFT_PEAK_MIN_HEIGHT = 0.01 # Amplitude mínima (em arcsec/s) para um pico ser considerado
FFT_PEAK_MIN_PROMINENCE = 0.01 # Quão um pico se destaca em relação aos vizinhos

# Constantes para Calibração Física
SIDEREAL_RATE_ARCSEC_S = 15.04108
SPEED_FACTOR = 4.0
THEORETICAL_SPEED_ARCSEC_S = SIDEREAL_RATE_ARCSEC_S * SPEED_FACTOR

# --- Estruturas de Dados ---
time_data = []
speed_data_raw_px_s = []
position_total_px = 0.0
calibration_factor_K = 0.0

# --- Setup dos Gráficos Interativos ---
plt.ion()
fig, (ax_speed, ax_fft_auto, ax_fft_fixed) = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
fig.tight_layout(pad=3.0)

# Gráfico 1: Velocidade
ax_speed.set_title('Velocidade de Deslocamento vs. Tempo (Filtrada)')
ax_speed.set_ylabel('Velocidade (arcsec/s)')
ax_speed.set_xlabel('Tempo (s)')
ax_speed.grid(True)
line_smooth_short, = ax_speed.plot([], [], '-', color='darkorange', linewidth=2, label=f'Média Móvel ({MOVING_AVERAGE_WINDOW_SHORT} pontos)')
line_smooth_long, = ax_speed.plot([], [], '-', color='crimson', linewidth=2, label=f'Média Móvel ({MOVING_AVERAGE_WINDOW_LONG} pontos)')
line_global_avg, = ax_speed.plot([], [], '--', color='cyan', linewidth=2, label='Média Global (Estável)')
ax_speed.legend()

# Gráfico 2: FFT com Escala Automática
ax_fft_auto.set_title('Análise de Frequência (FFT) - Escala Automática')
ax_fft_auto.set_ylabel('Amplitude (arcsec/s)')
ax_fft_auto.set_xlabel('Frequência (Hz)')
ax_fft_auto.grid(True)
line_fft_auto, = ax_fft_auto.plot([], [], '-', color='steelblue', linewidth=1, label='Espectro (Auto)')
# NOVO: Marcadores para os picos
peak_markers_auto, = ax_fft_auto.plot([], [], 'x', color='red', markersize=8, label='Picos Detectados')
ax_fft_auto.legend()
ax_fft_auto.set_xlim(0, 0.02)

# Gráfico 3: FFT com Escala Fixa
ax_fft_fixed.set_title('Análise de Frequência (FFT) - Região de Interesse')
ax_fft_fixed.set_xlabel('Frequência (Hz)')
ax_fft_fixed.set_ylabel('Amplitude (arcsec/s)')
ax_fft_fixed.grid(True)
line_fft_fixed, = ax_fft_fixed.plot([], [], '-', color='darkviolet', linewidth=1, label='Espectro (Fixo)')
# NOVO: Marcadores para os picos
peak_markers_fixed, = ax_fft_fixed.plot([], [], 'x', color='red', markersize=8) # Legenda já está no gráfico de cima
ax_fft_fixed.legend()
ax_fft_fixed.set_xlim(0, 0.02)
ax_fft_fixed.set_ylim(0, 0.5)

camera = None

def get_image_from_camera():
    global camera
    if camera is None: camera = cv2.VideoCapture(0)
    if not camera.isOpened(): print("Erro: Não foi possível abrir a câmera."); return None
    ret, frame = camera.read()
    if not ret: print("Erro: Não foi possível capturar o quadro."); return None
    return frame

def draw_flow(image_np, flow_tensor, magnitude_threshold=0.5):
    vis = image_np.copy()
    flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
    h, w = image_np.shape[:2]; step = 16
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow_np[y, x].T; magnitude = np.sqrt(fx**2 + fy**2); mask = magnitude > magnitude_threshold
    x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]
    if len(x) > 0:
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
        for (x1, y1), _ in lines: cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
    return vis

def update_plot():
    global calibration_factor_K
    if not time_data: return

    np_speed_data_px_s = np.array(speed_data_raw_px_s)
    np_time_data = np.array(time_data)
    np_speed_data_arcsec_s = np_speed_data_px_s * calibration_factor_K

    if len(np_speed_data_arcsec_s) >= MOVING_AVERAGE_WINDOW_SHORT:
        moving_avg_short = np.convolve(np_speed_data_arcsec_s, np.ones(MOVING_AVERAGE_WINDOW_SHORT), 'valid') / MOVING_AVERAGE_WINDOW_SHORT
        time_for_avg_short = np_time_data[MOVING_AVERAGE_WINDOW_SHORT-1:]
        line_smooth_short.set_data(time_for_avg_short, moving_avg_short)

    if len(np_speed_data_arcsec_s) >= MOVING_AVERAGE_WINDOW_LONG:
        moving_avg_long = np.convolve(np_speed_data_arcsec_s, np.ones(MOVING_AVERAGE_WINDOW_LONG), 'valid') / MOVING_AVERAGE_WINDOW_LONG
        time_for_avg_long = np_time_data[MOVING_AVERAGE_WINDOW_LONG-1:]
        line_smooth_long.set_data(time_for_avg_long, moving_avg_long)
        
        global_avg_values = np.cumsum(moving_avg_long) / (np.arange(len(moving_avg_long)) + 1)
        line_global_avg.set_data(time_for_avg_long, global_avg_values)
        
        if len(moving_avg_long) > 1:
            signal_sem_tendencia = detrend(moving_avg_long)
            N = len(signal_sem_tendencia)
            T = (time_for_avg_long[-1] - time_for_avg_long[0]) / (N - 1) if N > 1 else 1.0
            yf = np.abs(np.fft.rfft(signal_sem_tendencia)) / N * 2; yf[0] /= 2
            xf = np.fft.rfftfreq(N, T)
            line_fft_auto.set_data(xf, yf)
            line_fft_fixed.set_data(xf, yf)

            # --- NOVO: Lógica de Detecção de Picos ---
            peaks, _ = find_peaks(yf, height=FFT_PEAK_MIN_HEIGHT, prominence=FFT_PEAK_MIN_PROMINENCE)
            peak_freqs = xf[peaks]
            peak_amps = yf[peaks]
            peak_markers_auto.set_data(peak_freqs, peak_amps)
            peak_markers_fixed.set_data(peak_freqs, peak_amps)

            # Imprime os picos encontrados no console
            print("--- Picos FFT Detectados (Freq[Hz], Ampl[arcsec/s]) ---")
            if len(peak_freqs) > 0:
                for freq, amp in zip(peak_freqs, peak_amps):
                    print(f"  - Frequência: {freq:.5f} Hz (Período: {1/freq:.1f} s), Amplitude: {amp:.3f} arcsec/s")
            else:
                print("  - Nenhum pico significativo encontrado.")
            print("-" * 55)


    ax_speed.relim(); ax_speed.autoscale_view(True, True, True)
    ax_fft_auto.relim(); ax_fft_auto.autoscale_view(scalex=False, scaley=True)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def process_video_feed_with_raft():
    # (Lógica principal inalterada)
    print("Configurando o dispositivo e o modelo RAFT...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Processando no dispositivo: {device.upper()} ---")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval(); transforms = weights.transforms()
    global position_total_px, calibration_factor_K
    position_total_px = 0.0; calibration_factor_K = 0.0
    prev_time_fps = time.time(); last_plot_update = time.time()
    prev_frame_np = get_image_from_camera()
    if prev_frame_np is None: return "Câmera não iniciada"
    print("Câmera inicializada com sucesso.")
    start_msec = camera.get(cv2.CAP_PROP_POS_MSEC); prev_msec = start_msec
    print("\nIniciando o loop de processamento. Feche a janela do gráfico para sair.")
    
    while plt.fignum_exists(fig.number):
        current_time_fps = time.time()
        fps = 1.0 / (current_time_fps - prev_time_fps) if current_time_fps != prev_time_fps else 0
        prev_time_fps = current_time_fps
        current_frame_np = get_image_from_camera()
        if current_frame_np is None: break
        current_msec = camera.get(cv2.CAP_PROP_POS_MSEC)
        delta_time_s = (current_msec - prev_msec) / 1000.0 if current_msec != prev_msec else 1e-6
        prev_msec = current_msec
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
        if len(speed_data_raw_px_s) > MOVING_AVERAGE_WINDOW_SHORT:
            global_mean = np.mean(speed_data_raw_px_s); global_std = np.std(speed_data_raw_px_s)
            upper_threshold = global_mean + OUTLIER_STD_DEV_THRESHOLD * global_std
            lower_threshold = max(0, global_mean - OUTLIER_STD_DEV_THRESHOLD * global_std)
            if current_speed_px_s > upper_threshold or current_speed_px_s < lower_threshold:
                is_outlier = True
        
        if not is_outlier:
            elapsed_time = (current_msec - start_msec) / 1000.0
            time_data.append(elapsed_time)
            speed_data_raw_px_s.append(current_speed_px_s)
            position_total_px += current_speed_px_s * delta_time_s
            
            if len(speed_data_raw_px_s) >= MOVING_AVERAGE_WINDOW_LONG:
                latest_global_avg_px_s = np.mean(np.convolve(speed_data_raw_px_s, np.ones(MOVING_AVERAGE_WINDOW_LONG), 'valid') / MOVING_AVERAGE_WINDOW_LONG)
                if latest_global_avg_px_s > 0:
                    calibration_factor_K = THEORETICAL_SPEED_ARCSEC_S / latest_global_avg_px_s
        
        if time.time() - last_plot_update > PLOT_UPDATE_SECONDS:
            update_plot()
            last_plot_update = time.time()
        
        speed_now_arcsec_s = current_speed_px_s * calibration_factor_K
        position_total_arcsec = position_total_px * calibration_factor_K
        visualization = draw_flow(current_frame_np, predicted_flow[0])
        overlay = visualization.copy()
        cv2.rectangle(overlay, (5, 5), (380, 130), (0, 0, 0), -1); alpha = 0.6
        visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)
        cv2.putText(visualization, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(visualization, f"Flow Time: {flow_proc_time * 1000:.1f} ms", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(visualization, f"Speed Now: {speed_now_arcsec_s:.2f} arcsec/s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(visualization, f"Position: {position_total_arcsec:.1f} arcsec", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Fluxo Optico - RAFT em Tempo Real', visualization)
        
        plt.pause(1e-3)
        if cv2.getWindowProperty('Fluxo Optico - RAFT em Tempo Real', cv2.WND_PROP_VISIBLE) < 1: break
        prev_frame_np = current_frame_np.copy()

if __name__ == '__main__':
    try:
        process_video_feed_with_raft()
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        if camera is not None and camera.isOpened(): camera.release(); print("Câmera liberada.")
        plt.ioff(); plt.close('all'); print("Gráficos fechados.")
        cv2.destroyAllWindows(); print("Janelas fechadas. Programa encerrado.")