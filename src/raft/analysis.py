# src/raft/analysis.py

import numpy as np
from scipy.signal import detrend, find_peaks
from . import config

def process_analysis_data(time_data, speed_data_raw_px_s, calibration_factor_K, maf_short_points, maf_long_points):
    """
    Executa todos os cálculos de análise sobre os dados brutos.
    Recebe os tamanhos de janela de filtro como argumentos.
    """
    # Se os filtros ainda não foram inicializados, não faz nada
    if not time_data or not maf_long_points:
        return None

    # Garante que temos dados suficientes para o maior filtro
    if len(speed_data_raw_px_s) < maf_long_points:
        return None

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
    yf = np.abs(np.fft.rfft(signal_final)) / N * 2; yf[0] /= 2
    xf = np.fft.rfftfreq(N, T)
    
    peaks, _ = find_peaks(yf, height=config.FFT_PEAK_MIN_HEIGHT, prominence=config.FFT_PEAK_MIN_PROMINENCE)
    
    results = {
        "time_for_avg_short": time_for_avg_short, "moving_avg_short": moving_avg_short,
        "time_for_avg_long": time_for_avg_long, "moving_avg_long": moving_avg_long,
        "global_avg_values": global_avg_values, "fft_freqs": xf,
        "fft_amps": yf, "peak_indices": peaks
    }
    return results