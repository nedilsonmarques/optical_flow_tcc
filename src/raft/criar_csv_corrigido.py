# criar_csv_corrigido.py

import pandas as pd
import numpy as np
import sys
import os

def corrigir_csv_sessao(caminho_arquivo_antigo, caminho_arquivo_novo):
    """
    Lê um arquivo CSV de sessão com formato antigo (timestamp absoluto e velocidade errada),
    corrige os dados e salva em um novo arquivo CSV com o formato correto.
    """
    print(f"Lendo arquivo antigo: {caminho_arquivo_antigo}")
    try:
        df = pd.read_csv(caminho_arquivo_antigo)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado. Verifique o caminho.")
        return

    # --- Validação das Colunas ---
    # O formato antigo tinha a primeira coluna com timestamps absolutos, mas com o nome errado.
    # Vamos assumir que as colunas importantes são a primeira, a terceira e a quarta.
    if len(df.columns) < 4:
        print("ERRO: O arquivo CSV não parece ter o formato esperado (pelo menos 4 colunas).")
        return

    print("Formato antigo detectado. Iniciando correção...")
    coluna_tempo_antigo = df.columns[0]
    coluna_flow_x = df.columns[2]
    coluna_flow_y = df.columns[3]
    
    # --- 1. Corrigindo o Tempo ---
    timestamps = df[coluna_tempo_antigo].values
    tempo_decorrido_s = (timestamps - timestamps[0]) / 1000.0
    
    # --- 2. Recalculando a Velocidade ---
    delta_tempos_s = np.diff(tempo_decorrido_s, prepend=0)
    # Estima o primeiro delta_t como a mediana dos outros para evitar divisão por zero
    if len(delta_tempos_s) > 1:
        delta_tempos_s[0] = np.median(delta_tempos_s[1:])
    
    flow_x = df[coluna_flow_x].values
    flow_y = df[coluna_flow_y].values
    
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    velocidade_px_s = np.divide(magnitude, delta_tempos_s, out=np.zeros_like(magnitude), where=delta_tempos_s!=0)
    
    # --- 3. Criando o Novo DataFrame ---
    df_corrigido = pd.DataFrame({
        'elapsed_time_s': tempo_decorrido_s,
        'speed_px_s': velocidade_px_s,
        'flow_x_px': flow_x,
        'flow_y_px': flow_y
    })
    
    # --- 4. Salvando o Novo Arquivo ---
    try:
        df_corrigido.to_csv(caminho_arquivo_novo, index=False, float_format='%.4f')
        print(f"Sucesso! Arquivo corrigido salvo em: {caminho_arquivo_novo}")
    except Exception as e:
        print(f"ERRO ao salvar o novo arquivo: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python criar_csv_corrigido.py <caminho_do_arquivo_antigo> <caminho_do_arquivo_novo>")
        sys.exit(1)
        
    arquivo_antigo = sys.argv[1]
    arquivo_novo = sys.argv[2]
    
    corrigir_csv_sessao(arquivo_antigo, arquivo_novo)