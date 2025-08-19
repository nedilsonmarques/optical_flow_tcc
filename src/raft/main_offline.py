# src/raft/main_offline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import os

from . import config, analysis, visualization

def main():
    """
    Script principal para análise offline a partir de um arquivo de dados de sessão.
    """
    print(f"--- MODO: Análise Offline de Arquivo CSV ---")
    print(f"Carregando dados de: {config.SESSION_CSV_PATH}")

    try:
        df = pd.read_csv(config.SESSION_CSV_PATH)
        
        if 'elapsed_time_s' in df.columns and 'speed_px_s' in df.columns:
            print("Formato de CSV novo detectado.")
            time_data = df['elapsed_time_s'].values
            speed_data_raw_px_s = df['speed_px_s'].values
        elif 'timestamp_ms' in df.columns and 'flow_magnitude_px' in df.columns:
            print("AVISO: Formato de CSV antigo detectado. Convertendo dados...")
            timestamps = df['timestamp_ms'].values
            time_data = (timestamps - timestamps[0]) / 1000.0
            delta_times = np.diff(time_data, prepend=0)
            if len(delta_times) > 1:
                delta_times[0] = np.median(delta_times[1:])
            magnitudes = df['flow_magnitude_px'].values
            speed_data_raw_px_s = np.divide(magnitudes, delta_times, out=np.zeros_like(magnitudes), where=delta_times!=0)
        else:
            raise ValueError("Formato de CSV desconhecido. Colunas essenciais não encontradas.")

        invalid_indices = ~np.isfinite(speed_data_raw_px_s)
        if np.any(invalid_indices):
            print(f"AVISO: Encontrados e removidos {np.sum(invalid_indices)} pontos de dados inválidos (NaN ou Inf).")
            speed_data_raw_px_s = speed_data_raw_px_s[~invalid_indices]
            time_data = time_data[~invalid_indices]

    except FileNotFoundError:
        print(f"ERRO: Arquivo CSV não encontrado em '{config.SESSION_CSV_PATH}'."); return
    except Exception as e:
        print(f"ERRO: Não foi possível ler o arquivo CSV. Detalhes: {e}"); traceback.print_exc(); return

    print(f"Carregados {len(time_data)} pontos de dados válidos.")
    
    if len(time_data) < 2:
        print("ERRO: Não há dados suficientes para análise após a limpeza."); return

    avg_delta_s = np.median(np.diff(time_data))
    avg_fps = 1.0 / avg_delta_s if avg_delta_s > 0 else 0
    maf_short_points = max(1, int(config.MOVING_AVERAGE_SHORT_SECONDS * avg_fps))
    maf_long_points = max(1, int(config.MOVING_AVERAGE_LONG_SECONDS * avg_fps))
    print("-" * 50)
    print(f"FPS de captura (mediana): {avg_fps:.2f} ({(1/avg_fps)*1000 if avg_fps>0 else 0:.1f} ms/quadro)")
    print(f"Filtro de Média Móvel Curto definido para {maf_short_points} pontos.")
    print(f"Filtro de Média Móvel Longo definido para {maf_long_points} pontos.")
    print("-" * 50)
    
    stable_speed_px_s = np.mean(speed_data_raw_px_s[maf_long_points:]) if len(speed_data_raw_px_s) > maf_long_points else np.mean(speed_data_raw_px_s)
    calibration_factor_K = config.THEORETICAL_SPEED_ARCSEC_S / stable_speed_px_s if stable_speed_px_s > 0 else 0
    print(f"Fator de calibração K calculado: {calibration_factor_K:.4f} arcsec/pixel")

    print("Processando análise de dados...")
    analysis_results = analysis.process_analysis_data(time_data, speed_data_raw_px_s, calibration_factor_K, maf_short_points, maf_long_points)

    print("Gerando gráficos...")
    fig, axes, lines, annotations, harmonic_lines = visualization.create_plots()
    visualization.update_plots(fig, axes, lines, annotations, harmonic_lines, analysis_results)
    
    try:
        session_dir = os.path.dirname(config.SESSION_CSV_PATH)
        output_filepath = os.path.join(session_dir, "analise_graficos.png")
        fig.savefig(output_filepath, dpi=150, bbox_inches='tight')
        print("-" * 50)
        print(f"Gráficos salvos como imagem em: {output_filepath}")
        print("-" * 50)
    except Exception as e:
        print(f"AVISO: Não foi possível salvar a imagem dos gráficos. Erro: {e}")
    
    print("\nAnálise concluída. Exibindo gráficos.")
    print("Feche a janela do gráfico para encerrar o programa.")
    
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        traceback.print_exc()