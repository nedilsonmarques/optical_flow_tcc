# src/raft/config.py

"""
Arquivo de configuração central para todos os parâmetros do projeto.
"""
# --- Configurações de Fonte de Dados ---
# Altere para False para usar uma pasta com imagens salvas
USE_LIVE_CAMERA = False

# Se USE_LIVE_CAMERA for False, especifique o caminho para a pasta com as imagens.
IMAGE_FOLDER_PATH = "D:\\TCC\\tcc\\cap_img" 

# --- Configurações de Análise ---
PLOT_UPDATE_SECONDS = 1.0
# MODIFICADO: Filtros agora são definidos em segundos
MOVING_AVERAGE_SHORT_SECONDS = 10.0 # Média dos últimos 10 segundos
MOVING_AVERAGE_LONG_SECONDS = 50.0  # Média dos últimos 50 segundos
OUTLIER_STD_DEV_THRESHOLD = 3.0 
FFT_PEAK_MIN_HEIGHT = 0.01
FFT_PEAK_MIN_PROMINENCE = 0.01

# --- Constantes Físicas da Montagem ---
WORM_GEAR_TEETH = 144
SIDEREAL_DAY_S = 86164.09
SPEED_FACTOR = 4.0

# --- Cálculos Derivados ---
FUNDAMENTAL_FREQ_HZ = WORM_GEAR_TEETH / SIDEREAL_DAY_S
EXPECTED_PEAK_FREQ_HZ = FUNDAMENTAL_FREQ_HZ * SPEED_FACTOR
THEORETICAL_SPEED_ARCSEC_S = 15.04108 * SPEED_FACTOR