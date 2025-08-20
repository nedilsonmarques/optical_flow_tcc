# src/raft/config.py

"""
Arquivo de configuração central para todos os parâmetros do projeto.
"""
# --- Configurações de Fonte de Dados ---
USE_LIVE_CAMERA = False
#IMAGE_FOLDER_PATH = "D:\\TCC\\tcc\\cap_img\\" 
IMAGE_FOLDER_PATH = "D:\\RAFT\\sessions\\session_2025-08-17_11-11-55\\images\\" 

# --- Configurações de Salvamento de Sessão ---
SESSIONS_BASE_FOLDER = "D:\\RAFT\\sessions" # Pasta para salvar os arquivos CSV

# --- Configurações de Análise ---
PLOT_UPDATE_SECONDS = 1.0
MOVING_AVERAGE_SHORT_SECONDS = 10.0
MOVING_AVERAGE_LONG_SECONDS = 50.0
OUTLIER_STD_DEV_THRESHOLD = 3.0 
FFT_PEAK_MIN_HEIGHT = 0.01
FFT_PEAK_MIN_PROMINENCE = 0.01
FFT_MAX_PEAKS_TO_DISPLAY = 5    

# --- Constantes Físicas da Montagem ---
WORM_GEAR_TEETH = 144
SIDEREAL_DAY_S = 86164.09
SPEED_FACTOR = 4.0

# --- Cálculos Derivados ---
FUNDAMENTAL_FREQ_HZ = WORM_GEAR_TEETH / SIDEREAL_DAY_S
EXPECTED_PEAK_FREQ_HZ = FUNDAMENTAL_FREQ_HZ * SPEED_FACTOR
THEORETICAL_SPEED_ARCSEC_S = 15.04108 * SPEED_FACTOR

# --- Configurações de Análise Offline ---
# Coloque aqui o caminho para o arquivo CSV da sessão que você quer analisar.
SESSION_CSV_PATH = "D:\\RAFT\\sessions\\session_2025-08-18_23-33-32\\data.csv"