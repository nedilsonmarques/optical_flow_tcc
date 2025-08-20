# Análise de Erro Periódico de Montagem Equatorial com Fluxo Óptico

Este projeto utiliza técnicas de visão computacional, especificamente o fluxo óptico, para medir e analisar com alta precisão o erro periódico e outras anomalias de movimento no eixo de Ascenção Reta (RA) de uma montagem de telescópio.

O sistema foi desenvolvido para caracterizar uma montagem Meade LXD55, usando um microscópio USB para capturar imagens de uma superfície acoplada ao eixo RA. Os dados de movimento são processados em tempo real para fornecer uma análise detalhada da performance da montagem em unidades físicas (arcosegundos).

## Funcionalidades

* **Medição de Deslocamento:** Utiliza o algoritmo de fluxo óptico de última geração **RAFT** (via PyTorch) com aceleração por GPU (CUDA) para medições de sub-pixel.
* **Análise em Tempo Real:** Calcula velocidade, posição e aplica filtros estatísticos para rejeição de outliers.
* **Análise de Frequência Avançada:** Aplica uma cadeia de processamento de sinais (Detrending, Janela de Hanning) para gerar um Espectro de Frequência (FFT) limpo, identificando os componentes do erro periódico.
* **Visualização Dinâmica:** Apresenta os dados em um dashboard com múltiplos gráficos (Velocidade vs. Tempo, FFT, etc.) usando Matplotlib.
* **Calibração Física:** Converte as medições de `pixels/s` para `arcosegundos/s` através de uma calibração automática baseada na velocidade teórica da montagem.
* **Gerenciamento de Sessão:** Salva automaticamente todos os dados brutos (imagens e CSV de velocidade) e metadados (parâmetros de análise) em pastas de sessão únicas para cada execução.
* **Análise Offline:** Inclui um script para carregar, reprocessar e analisar sessões de dados previamente salvas.

## Estrutura do Projeto

O código foi refatorado em uma arquitetura modular para maior clareza e manutenibilidade, localizado em `src/raft/`.

* **`config.py`**: Arquivo central para todas as constantes e parâmetros de configuração, como caminhos de arquivo, parâmetros físicos da montagem e limiares de análise.
* **`session_manager.py`**: Módulo responsável por criar e gerenciar as sessões de captura. Cuida da criação de pastas, salvamento de metadados (`metadata.json`), dados de velocidade (`data.csv`) e imagens.
* **`analysis.py`**: O "cérebro" da análise. Contém toda a lógica de processamento numérico (médias móveis, FFT, detecção de picos, etc.). Não possui código de visualização.
* **`visualization.py`**: Responsável por toda a saída visual. Cria e atualiza os gráficos do Matplotlib e desenha o quadro de informações (HUD) no vídeo do OpenCV.
* **`main_live.py`**: Ponto de entrada para a **aquisição de dados** (seja da câmera ao vivo ou do reprocessamento de uma pasta de imagens). Orquestra os outros módulos para capturar, analisar, visualizar e salvar os dados em tempo real.
* **`main_offline.py`**: Ponto de entrada para a **análise de dados já salvos**. Carrega um arquivo `data.csv` de uma sessão anterior e gera os gráficos finais.

## Instalação e Execução

### 1. Requisitos

O projeto utiliza Python 3.11+. As dependências estão listadas no arquivo `requirements.txt`.

### 2. Configuração do Ambiente

É altamente recomendado usar um ambiente virtual.

```bash
# 1. Navegue até a pasta raiz do projeto (D:\RAFT)
cd D:\RAFT

# 2. Crie um ambiente virtual
python -m venv .venv

# 3. Ative o ambiente virtual
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# No Linux/macOS:
# source .venv/bin/activate
```

### 3. Instalação das Dependências

Com o ambiente ativado, instale os pacotes necessários:

```bash
pip install -r requirements.txt
```
**Nota:** O arquivo `requirements.txt` está configurado para instalar a versão do PyTorch com suporte a **CUDA 12.1**. Certifique-se de que seus drivers NVIDIA sejam compatíveis.

### 4. Execução

Todos os scripts devem ser executados como módulos a partir da pasta raiz do projeto (`D:\RAFT`).

* **Para Análise em Tempo Real (Câmera ou Pasta de Imagens):**
    * Configure os parâmetros desejados em `src/raft/config.py` (`USE_LIVE_CAMERA`, `IMAGE_FOLDER_PATH`, etc.).
    * Execute o comando:
    ```bash
    python -m src.raft.main_live
    ```

* **Para Análise de um CSV Salvo:**
    * Configure o caminho do arquivo em `src/raft/config.py` (variável `SESSION_CSV_PATH`).
    * Execute o comando:
    ```bash
    python -m src.raft.main_offline
    ```