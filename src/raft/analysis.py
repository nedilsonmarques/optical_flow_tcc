# src/raft/analysis.py

import numpy as np
from scipy.signal import detrend, find_peaks
from . import config

def process_analysis_data(time_data, speed_data_raw_px_s, calibration_factor_K, maf_short_points, maf_long_points):
    """
    Executa todos os cálculos de análise sobre os dados brutos.
    Recebe os tamanhos de janela de filtro como argumentos.
    """
    # Checagem robusta que funciona para listas e arrays
    if len(time_data) == 0 or maf_long_points is None or len(speed_data_raw_px_s) < maf_long_points:
        return None

    # Garante que os dados sejam arrays numpy para os cálculos
    np_speed_data_px_s = np.array(speed_data_raw_px_s)
    np_time_data = np.array(time_data)
    
    np_speed_data_arcsec_s = np_speed_data_px_s * calibration_factor_K
    np_speed_data_normalized = np_speed_data_arcsec_s / config.SPEED_FACTOR
    
    moving_avg_short = np.convolve(np_speed_data_normalized, np.ones(maf_short_points), 'valid') / maf_short_points
    moving_avg_long = np.convolve(np_speed_data_normalized, np.ones(maf_long_points), 'valid') / maf_long_points
    global_avg_values = np.cumsum(moving_avg_long) / (np.arange(len(moving_avg_long)) + 1)
    
    time_for_avg_short = np_time_data[maf_short_points-1:]
    time_for_avg_long = np_time_data[maf_long_points-1:]
    
    signal_sem_tendencia = detrend(moving_avg_long)
    window = np.hanning(len(signal_sem_tendencia))
    signal_final = signal_sem_tendencia * window
    
    N = len(signal_final)
    T = (time_for_avg_long[-1] - time_for_avg_long[0]) / (N - 1) if N > 1 else 1.0
    yf = np.abs(np.fft.rfft(signal_final)) / N * 2
    if len(yf) > 0:
        yf[0] = yf[0] / 2
    
    xf = np.fft.rfftfreq(N, T)
    
    all_peaks, properties = find_peaks(yf, height=config.FFT_PEAK_MIN_HEIGHT, prominence=config.FFT_PEAK_MIN_PROMINENCE)
    
    top_n_peak_indices = np.array([], dtype=int)
    if all_peaks.size > 0:
        peak_prominences = properties['prominences']
        peak_data = sorted(zip(all_peaks, peak_prominences), key=lambda x: x[1], reverse=True)
        top_n_peaks_unorded = [index for index, prominence in peak_data[:config.FFT_MAX_PEAKS_TO_DISPLAY]]
        top_n_peak_indices = np.sort(top_n_peaks_unorded)

    results = {
        "time_for_avg_short": time_for_avg_short, "moving_avg_short": moving_avg_short,
        "time_for_avg_long": time_for_avg_long, "moving_avg_long": moving_avg_long,
        "global_avg_values": global_avg_values, "fft_freqs": xf,
        "fft_amps": yf, 
        "peak_indices": top_n_peak_indices
    }
    return results