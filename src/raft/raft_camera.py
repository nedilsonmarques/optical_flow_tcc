import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import time # Importa a biblioteca de tempo

# --- Variáveis Globais ---
camera = None

def get_image_from_camera():
    """
    Captura e retorna um único quadro da câmera padrão do computador.
    Inicializa a câmera na primeira chamada.
    """
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
    """
    Desenha os vetores de fluxo óptico em uma imagem.
    """
    vis = image_np.copy()
    flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
    h, w = image_np.shape[:2]
    step = 16
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow_np[y, x].T
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > magnitude_threshold
    x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]

    if len(x) == 0:
        return vis

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
    for (x1, y1), _ in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
        
    return vis

def process_video_feed_with_raft():
    """
    Função principal que executa o loop de processamento de vídeo com métricas.
    """
    print("Configurando o dispositivo e o modelo RAFT...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Processando no dispositivo: {device.upper()} ---")

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval()

    transforms = weights.transforms()
    
    # --- NOVO: Inicialização de variáveis para métricas ---
    frame_counter = 0
    speed_accumulator = 0.0
    displayed_avg_speed = 0.0
    prev_time = time.time() # Tempo inicial para cálculo do primeiro FPS

    prev_frame_np = get_image_from_camera()
    if prev_frame_np is None:
        return

    print("\nIniciando o loop de processamento. Pressione 'q' na janela para sair.")
    
    while True:
        # --- NOVO: Cálculo de tempo e FPS ---
        current_time = time.time()
        delta_time = current_time - prev_time
        # Evita divisão por zero se o tempo for muito curto
        fps = 1.0 / delta_time if delta_time > 0 else 0
        prev_time = current_time

        current_frame_np = get_image_from_camera()
        if current_frame_np is None:
            break

        prev_frame_tensor = F.to_tensor(prev_frame_np).unsqueeze(0)
        current_frame_tensor = F.to_tensor(current_frame_np).unsqueeze(0)
        
        img1_batch, img2_batch = transforms(prev_frame_tensor, current_frame_tensor)
        img1_batch, img2_batch = img1_batch.to(device), img2_batch.to(device)

        # --- NOVO: Medição do tempo de processamento do fluxo ---
        flow_start_time = time.time()
        with torch.no_grad():
            list_of_flows = model(img1_batch, img2_batch)
        flow_proc_time = time.time() - flow_start_time
        
        predicted_flow = list_of_flows[-1] # Mantém o fluxo na GPU para o cálculo de velocidade

        # --- NOVO: Cálculo de Velocidade Média ---
        # Calcula a magnitude média do deslocamento (em pixels)
        magnitude = torch.sqrt(predicted_flow[0, 0]**2 + predicted_flow[0, 1]**2)
        avg_magnitude_px = magnitude.mean().item() # .item() para obter um float Python
        
        # Calcula a velocidade atual (pixels/segundo)
        current_speed_px_per_s = avg_magnitude_px / delta_time if delta_time > 0 else 0.0
        
        # Acumula valores para a média
        speed_accumulator += current_speed_px_per_s
        frame_counter += 1
        
        # Atualiza a velocidade média exibida a cada 30 quadros
        if frame_counter >= 30:
            displayed_avg_speed = speed_accumulator / frame_counter
            # Reinicia para o próximo ciclo de medição
            frame_counter = 0
            speed_accumulator = 0.0
        
        # Cria a visualização (passando o fluxo para a CPU)
        visualization = draw_flow(current_frame_np, predicted_flow[0])
        
        # --- NOVO: Adiciona o texto das métricas na tela ---
        # Fundo semi-transparente para o texto
        overlay = visualization.copy()
        cv2.rectangle(overlay, (5, 5), (290, 100), (0, 0, 0), -1)
        alpha = 0.6
        visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)
        
        # Textos
        fps_text = f"FPS: {fps:.1f}"
        time_text = f"Flow Time: {flow_proc_time * 1000:.1f} ms"
        speed_text = f"Avg Speed: {displayed_avg_speed:.1f} pixels/s"
        
        cv2.putText(visualization, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, speed_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Fluxo Optico - RAFT com Metricas', visualization)
        
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
        cv2.destroyAllWindows()
        print("Janelas fechadas. Programa encerrado.")