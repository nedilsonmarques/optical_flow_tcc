import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import numpy as np
import cv2

def get_image():
    """
    Gerador que simula a captura de imagens sucessivas.
    Produz uma sequência de imagens 480x640 com um quadrado branco se movendo.
    """
    height, width = 480, 640
    box_size = 50
    # Gera 100 quadros para o exemplo
    for i in range(100):
        # Cria uma imagem preta
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Posição inicial e final do quadrado
        start_x = 10 + i * 5
        start_y = 100
        
        # Garante que o quadrado não saia da tela
        if start_x + box_size > width:
            break
            
        # Desenha um quadrado branco na imagem
        frame[start_y:start_y + box_size, start_x:start_x + box_size] = [255, 255, 255]
        
        yield frame

# Função para visualizar o fluxo óptico (diferente da anterior para lidar com o formato do tensor)

# CORREÇÃO NA FUNÇÃO DE VISUALIZAÇÃO
def draw_flow_from_tensor(image_np, flow_tensor):
    """
    Desenha o fluxo óptico em uma imagem a partir de um tensor PyTorch.
    """
    # CORREÇÃO: A linha abaixo foi alterada. Usamos .copy() para criar uma cópia.
    vis = image_np.copy()
    
    # Converte o tensor de fluxo (C, H, W) para um array numpy (H, W, C)
    flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
    
    h, w = image_np.shape[:2]
    step = 16 # Desenha um vetor a cada 16 pixels
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    fx, fy = flow_np[y, x].T
    
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Desenha as linhas (vetores de fluxo) em verde
    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
    
    # Desenha os pontos de origem dos vetores em vermelho
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
        
    return vis

# def draw_flow_from_tensor(image_np, flow_tensor):
#     # Converte o tensor de fluxo (C, H, W) para um array numpy (H, W, C)
#     flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)
    
#     h, w = image_np.shape[:2]
#     step = 16
#     y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
#     fx, fy = flow_np[y, x].T
    
#     lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
    
#     vis = cv2.cvtColor(image_np, cv2.COLOR_BGR2BGR)
#     cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=1)
    
#     for (x1, y1), (_x2, _y2) in lines:
#         cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
#     return vis


def process_with_pytorch_raft():
    """
    Calcula e exibe o fluxo óptico usando o modelo RAFT com PyTorch.
    """
    # Verifica se a GPU está disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Iniciando com PyTorch (RAFT) no dispositivo: {device} ---")

    # Carrega o modelo RAFT pré-treinado e o move para o dispositivo (GPU ou CPU)
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device)
    model = model.eval() # Coloca o modelo em modo de avaliação

    # Obtém as transformações necessárias para o modelo
    transforms = weights.transforms()

    image_generator = get_image()
    
    # Pega o primeiro quadro
    prev_frame_np = next(image_generator, None)
    if prev_frame_np is None:
        return

    for current_frame_np in image_generator:
        # Converte os frames numpy (H, W, C) para tensores (C, H, W)
        prev_frame_tensor = F.to_tensor(prev_frame_np).unsqueeze(0)
        current_frame_tensor = F.to_tensor(current_frame_np).unsqueeze(0)
        
        # CORREÇÃO: Aplicar as transformações aos DOIS frames JUNTOS.
        # A função de transformação retorna os dois tensores já processados.
        img1_batch, img2_batch = transforms(prev_frame_tensor, current_frame_tensor)

        # CORREÇÃO: Mover os tensores para o dispositivo APÓS a transformação.
        img1_batch = img1_batch.to(device)
        img2_batch = img2_batch.to(device)
        
        # Aplica as transformações e move os tensores para o dispositivo
        # img1_batch = transforms(prev_frame_tensor).to(device)
        # img2_batch = transforms(current_frame_tensor).to(device)

        # Desativa o cálculo de gradientes para acelerar a inferência
        with torch.no_grad():
            # O modelo retorna uma lista de previsões de fluxo; a última é a mais refinada
            list_of_flows = model(img1_batch, img2_batch)
            predicted_flow = list_of_flows[-1][0] # Pega o fluxo do primeiro item do batch

        # Visualiza o resultado
        visualization = draw_flow_from_tensor(current_frame_np, predicted_flow)
        cv2.imshow('Optical Flow - PyTorch RAFT', visualization)
        
        # if cv2.waitKey(30) & 0xff == ord('q'):
        #     break
        if cv2.waitKey(1) & 0xff == ord('q'):
             break
        
        # Atualiza o quadro anterior
        prev_frame_np = current_frame_np
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Descomente a solução que deseja executar

    # --- Solução 1: OpenCV ---
    # USE_GPU_OPENCV = True 
    # print("--- Iniciando com OpenCV ---")
    # process_with_opencv(use_gpu=USE_GPU_OPENCV)
    
    # --- Solução 2: PyTorch RAFT ---
    process_with_pytorch_raft()