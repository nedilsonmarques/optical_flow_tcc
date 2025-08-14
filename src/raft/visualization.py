# src/raft/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
from . import config

def create_plots():
    # (Esta função não precisa de alterações)
    plt.ion()
    fig, (ax_speed, ax_fft_wide, ax_fft_zoom) = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    fig.tight_layout(pad=3.0)
    ax_speed.set_title('Velocidade Sideral Equivalente (1x) vs. Tempo')
    ax_speed.set_ylabel('Velocidade (arcsec/s @ 1x)')
    ax_speed.set_xlabel('Tempo (s)')
    ax_speed.grid(True)
    line_smooth_short, = ax_speed.plot([], [], '-', color='darkorange', linewidth=2, label=f'Média Móvel ({config.MOVING_AVERAGE_SHORT_SECONDS:.0f}s)')
    line_smooth_long, = ax_speed.plot([], [], '-', color='crimson', linewidth=2, label=f'Média Móvel ({config.MOVING_AVERAGE_LONG_SECONDS:.0f}s)')
    line_global_avg, = ax_speed.plot([], [], '--', color='cyan', linewidth=2, label='Média Global (Estável)')
    ax_speed.legend()
    ax_fft_wide.set_title('Análise de Frequência (FFT) - Visão Ampla')
    ax_fft_wide.set_ylabel('Amplitude (arcsec/s @ 1x)')
    ax_fft_wide.set_xlabel('Frequência (Hz)')
    ax_fft_wide.grid(True)
    line_fft_wide, = ax_fft_wide.plot([], [], '-', color='steelblue', linewidth=1, label='Espectro (Amplo)')
    peak_markers_wide, = ax_fft_wide.plot([], [], 'x', color='red', markersize=8, label='Picos Detectados')
    ax_fft_wide.legend()
    ax_fft_wide.set_xlim(0, 16 * config.EXPECTED_PEAK_FREQ_HZ)
    ax_fft_zoom.set_title('Análise de Frequência (FFT) - Pico Principal')
    ax_fft_zoom.set_xlabel('Frequência (Hz)')
    ax_fft_zoom.set_ylabel('Amplitude (arcsec/s @ 1x)')
    ax_fft_zoom.grid(True)
    line_fft_zoom, = ax_fft_zoom.plot([], [], '-', color='darkviolet', linewidth=1, label='Espectro (Zoom)')
    peak_markers_zoom, = ax_fft_zoom.plot([], [], 'x', color='red', markersize=8)
    ax_fft_zoom.legend()
    ax_fft_zoom.set_xlim(0, 2 * config.EXPECTED_PEAK_FREQ_HZ)
    lines = {"smooth_short": line_smooth_short, "smooth_long": line_smooth_long, "global_avg": line_global_avg, "fft_wide": line_fft_wide, "fft_zoom": line_fft_zoom, "peaks_wide": peak_markers_wide, "peaks_zoom": peak_markers_zoom}
    annotations = {"wide": [], "zoom": []}; harmonic_lines = {"wide": [], "zoom": []}
    return fig, (ax_speed, ax_fft_wide, ax_fft_zoom), lines, annotations, harmonic_lines

def update_plots(fig, axes, lines, annotations, harmonic_lines, analysis_results):
    # (Esta função não precisa de alterações)
    if analysis_results is None: return
    ax_speed, ax_fft_wide, ax_fft_zoom = axes
    lines["smooth_short"].set_data(analysis_results["time_for_avg_short"], analysis_results["moving_avg_short"])
    lines["smooth_long"].set_data(analysis_results["time_for_avg_long"], analysis_results["moving_avg_long"])
    lines["global_avg"].set_data(analysis_results["time_for_avg_long"], analysis_results["global_avg_values"])
    xf = analysis_results["fft_freqs"]; yf = analysis_results["fft_amps"]
    lines["fft_wide"].set_data(xf, yf); lines["fft_zoom"].set_data(xf, yf)
    peak_indices = analysis_results["peak_indices"]
    peak_freqs = xf[peak_indices]; peak_amps = yf[peak_indices]
    lines["peaks_wide"].set_data(peak_freqs, peak_amps); lines["peaks_zoom"].set_data(peak_freqs, peak_amps)
    for ann in annotations["wide"]: ann.remove()
    for ann in annotations["zoom"]: ann.remove()
    annotations["wide"].clear(); annotations["zoom"].clear()
    print("--- Picos FFT Detectados (Freq[Hz], Ampl[arcsec/s @ 1x]) ---")
    if len(peak_freqs) > 0:
        for freq, amp in zip(peak_freqs, peak_amps):
            print(f"  - Frequência: {freq:.5f} Hz (Período: {1/freq:.1f} s), Amplitude: {amp:.3f} arcsec/s")
            text = f"({freq:.4f}, {amp:.3f})"
            ann_wide = ax_fft_wide.annotate(text, (freq, amp), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            annotations["wide"].append(ann_wide)
            ann_zoom = ax_fft_zoom.annotate(text, (freq, amp), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='darkviolet')
            annotations["zoom"].append(ann_zoom)
    else: print("  - Nenhum pico significativo encontrado.")
    print("-" * 60)
    for line in harmonic_lines["wide"]: line.remove()
    for line in harmonic_lines["zoom"]: line.remove()
    harmonic_lines["wide"].clear(); harmonic_lines["zoom"].clear()
    for i in range(1, 20):
        harmonic_freq = i * config.EXPECTED_PEAK_FREQ_HZ
        is_fundamental = (i == 1); color = 'green' if is_fundamental else 'gray'; style = '-' if is_fundamental else '--'
        if harmonic_freq < ax_fft_wide.get_xlim()[1]:
            line = ax_fft_wide.axvline(x=harmonic_freq, color=color, linestyle=style, linewidth=0.8, zorder=0)
            harmonic_lines["wide"].append(line)
        if harmonic_freq < ax_fft_zoom.get_xlim()[1]:
            line = ax_fft_zoom.axvline(x=harmonic_freq, color=color, linestyle=style, linewidth=0.8, zorder=0)
            harmonic_lines["zoom"].append(line)
    for ax in axes:
        ax.relim()
        if ax == ax_speed: ax.autoscale_view(True, True, True)
        else: ax.autoscale_view(scalex=False, scaley=True)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def draw_hud(frame, metrics):
    # (Esta função não precisa de alterações)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (420, 180), (0, 0, 0), -1); alpha = 0.6
    vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    y_start = 30; line_height = 25; font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; color = (0, 255, 255); thickness = 2
    pos_text = f"Posicao Total: {metrics['pos_total_arcsec']:.1f} arcsec"
    delta_text = f"Intervalo Frames (dt): {metrics['delta_time_ms']:.1f} ms"
    speed_text = f"Vel. Sideral (1x Eq): {metrics['speed_norm_arcsec_s']:.2f} arcsec/s"
    cv2.putText(vis, pos_text, (10, y_start), font, font_scale, color, thickness)
    cv2.putText(vis, delta_text, (10, y_start + line_height), font, font_scale, color, thickness)
    cv2.putText(vis, speed_text, (10, y_start + 2*line_height), font, font_scale, color, thickness)
    separator = "------------------------------------"
    cv2.putText(vis, separator, (10, y_start + 3*line_height), font, 0.5, (128, 128, 128), 1)
    fps_text = f"FPS (Processamento): {metrics['fps']:.1f}"
    infer_text = f"Tempo Inferencia: {metrics['flow_time_ms']:.1f} ms"
    cv2.putText(vis, fps_text, (10, y_start + 4*line_height), font, font_scale, color, thickness)
    cv2.putText(vis, infer_text, (10, y_start + 5*line_height), font, font_scale, color, thickness)
    return vis

def draw_flow(frame, flow_tensor, magnitude_threshold=0.5):
    """MODIFICADO: Corrige o desempacotamento do .shape para imagens coloridas."""
    vis = frame.copy()
    flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # CORREÇÃO: Pega apenas os 2 primeiros valores (altura, largura) do shape
    h, w, _ = frame.shape
    
    step = 16
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    # Garante que os índices não ultrapassem os limites do array de fluxo
    y = np.clip(y, 0, flow_np.shape[0] - 1)
    x = np.clip(x, 0, flow_np.shape[1] - 1)
    
    fx, fy = flow_np[y, x].T
    
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
    for (x1, y1), _ in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
    return vis