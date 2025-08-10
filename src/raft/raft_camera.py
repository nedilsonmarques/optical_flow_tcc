import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import time
import matplotlib.pyplot as plt
from collections import deque

# --- Variáveis Globais ---
camera = None

# --- NOVO: Configurações do Gráfico ---
PLOT_UPDATE_SECONDS = 1.0  # Atualiza o gráfico a cada 1 segundo
PLOT_WINDOW_SECONDS = 60   # O gráfico mostrará os últimos 60 segundos de dados
MOVING_AVERAGE_WINDOW = 30 # A média móvel será calculada sobre os últimos 30 pontos

# --- NOVO: Estruturas de Dados para o Gráfico ---
# Usamos deque com maxlen para criar uma janela de dados deslizante automática
# Estimamos um maxlen baseado em 60 FPS por segurança
data_buffer_size = PLOT_WINDOW_SECONDS * 60 
time_data = deque(maxlen=data_buffer_size)
speed_data_raw = deque(maxlen=data_buffer_size)

# --- NOVO: Setup do Gráfico Interativo ---
plt.ion()  # Habilita o modo interativo do Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
line_raw, = ax.plot([], [], 'o', color='dodgerblue', alpha=0.5, markersize=2, label='Velocidade Instantânea')
line_smooth, = ax.plot([], [], '-', color='darkorange', linewidth=2, label=f'Média Móvel ({MOVING_AVERAGE_WINDOW} pontos)')
ax.set_title('Velocidade Média de Deslocamento vs. Tempo')
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Velocidade (pixels/s)')
ax.legend()
ax.grid(True)


def get_image_from_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            return None
        print("Câmera inicializada com sucesso.")
    ret, frame = camera.read()
    if not ret:
        print("Erro: Não foi possível capturar o quadro.")
        return None
    return frame

def draw_flow(image_np, flow_tensor, magnitude_threshold=0.5):
    # (Esta função permanece inalterada)
    vis = image_np.copy()
    flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
    h, w = image_np.shape[:2]
    step = 16
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow_np[y, x].T
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > magnitude_threshold
    x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]
    if len(x) > 0:
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
        for (x1, y1), _ in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
    return vis

def update_plot():
    """ NOVO: Função para calcular a média móvel e redesenhar o gráfico. """
    if not time_data:
        return

    # Calcula a média móvel
    smooth_speeds = []
    if len(speed_data_raw) >= MOVING_AVERAGE_WINDOW:
        # Usamos uma visualização em janela deslizante para calcular a média
        moving_avg_data = list(np.convolve(list(speed_data_raw), np.ones(MOVING_AVERAGE_WINDOW), 'valid') / MOVING_AVERAGE_WINDOW)
        # Ajustamos o tempo para o centro da janela da média
        time_for_smooth = list(time_data)[MOVING_AVERAGE_WINDOW-1:]
        line_smooth.set_data(time_for_smooth, moving_avg_data)
    else:
        # Se não tivermos dados suficientes, não plotamos a média
        line_smooth.set_data([], [])

    # Atualiza os dados dos pontos brutos
    line_raw.set_data(list(time_data), list(speed_data_raw))
    
    # Reajusta os limites do gráfico (eixo y automático, eixo x como janela deslizante)
    current_max_time = time_data[-1]
    ax.set_xlim(max(0, current_max_time - PLOT_WINDOW_SECONDS), current_max_time + 1)
    ax.relim()
    ax.autoscale_view(True, True, True)
    
    # Força o redesenho do canvas
    fig.canvas.draw()
    fig.canvas.flush_events()


def process_video_feed_with_raft():
    """ Função principal que executa o loop de processamento de vídeo com métricas e gráfico. """
    print("Configurando o dispositivo e o modelo RAFT...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Processando no dispositivo: {device.upper()} ---")

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval()

    transforms = weights.transforms()
    
    # Variáveis para métricas
    prev_time = time.time()
    start_time = time.time()  # Tempo de início absoluto para o eixo X do gráfico
    last_plot_update = start_time # Tempo da última atualização do gráfico

    prev_frame_np = get_image_from_camera()
    if prev_frame_np is None:
        return

    print("\nIniciando o loop de processamento. Pressione 'q' na janela para sair.")
    
    while True:
        current_time = time.time()
        delta_time = current_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0
        prev_time = current_time

        current_frame_np = get_image_from_camera()
        if current_frame_np is None:
            break

        prev_frame_tensor = F.to_tensor(prev_frame_np).unsqueeze(0)
        current_frame_tensor = F.to_tensor(current_frame_np).unsqueeze(0)
        img1_batch, img2_batch = transforms(prev_frame_tensor, current_frame_tensor)
        img1_batch, img2_batch = img1_batch.to(device), img2_batch.to(device)

        with torch.no_grad():
            list_of_flows = model(img1_batch, img2_batch)
        
        predicted_flow = list_of_flows[-1]
        magnitude = torch.sqrt(predicted_flow[0, 0]**2 + predicted_flow[0, 1]**2)
        avg_magnitude_px = magnitude.mean().item()
        current_speed_px_per_s = avg_magnitude_px / delta_time if delta_time > 0 else 0.0
        
        # --- NOVO: Adiciona dados para o gráfico ---
        elapsed_time = current_time - start_time
        time_data.append(elapsed_time)
        speed_data_raw.append(current_speed_px_per_s)
        
        # --- NOVO: Atualiza o gráfico periodicamente ---
        if current_time - last_plot_update > PLOT_UPDATE_SECONDS:
            update_plot()
            last_plot_update = current_time

        visualization = draw_flow(current_frame_np, predicted_flow[0])
        
        # Exibição das métricas de texto (FPS e Velocidade)
        fps_text = f"FPS: {fps:.1f}"
        speed_text = f"Speed Now: {current_speed_px_per_s:.1f} pixels/s"
        cv2.putText(visualization, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(visualization, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Fluxo Optico - RAFT em Tempo Real', visualization)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame_np = current_frame_np

if __name__ == '__main__':
    try:
        process_video_feed_with_raft()
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        if camera is not None and camera.isOpened():
            camera.release()
            print("Câmera liberada.")
        
        # --- NOVO: Gerenciamento da janela do gráfico ---
        plt.ioff() # Desliga o modo interativo
        plt.close(fig) # Fecha a figura do gráfico
        print("Gráfico fechado.")
        
        cv2.destroyAllWindows()
        print("Janelas fechadas. Programa encerrado.")