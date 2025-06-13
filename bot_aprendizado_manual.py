import cv2
import numpy as np
import pygetwindow as gw
import mss
import os
import uuid
import time

# CONFIGURAÇÃO
NOME_JANELA = "Albion Online Client"

# Diretórios
TEMPLATE_DIR = r"C:\Users\augusto\Documents\albionPy"
MOBS_DIR = os.path.join(TEMPLATE_DIR, "mobs")
COLETAS_DIR = os.path.join(TEMPLATE_DIR, "coletas")
MOB_COLETA_DIR = os.path.join(TEMPLATE_DIR, "mob_coleta")
IGNORADOS_DIR = os.path.join(TEMPLATE_DIR, "ignorados")
TEMP_DIR = os.path.join(TEMPLATE_DIR, "temp")

# Criação das pastas
for pasta in [MOBS_DIR, COLETAS_DIR, MOB_COLETA_DIR, IGNORADOS_DIR, TEMP_DIR]:
    os.makedirs(pasta, exist_ok=True)

# Captura da tela da janela com mss
def capturar_tela_mss():
    try:
        janela = gw.getWindowsWithTitle(NOME_JANELA)[0]
        if not janela.isMinimized:
            monitor = {
                "top": janela.top,
                "left": janela.left,
                "width": janela.width,
                "height": janela.height
            }
            with mss.mss() as sct:
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            print("Janela minimizada.")
            return None
    except IndexError:
        print(f"Janela '{NOME_JANELA}' não encontrada.")
        return None

# Carrega templates já classificados
def carregar_templates(pasta):
    templates = []
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        img = cv2.imread(caminho)
        if img is not None:
            templates.append((arquivo, img))
    return templates

# Verifica se o recorte já é conhecido
def ja_conhecido(recorte, conhecidos, threshold=0.95):
    for nome, template in conhecidos:
        if template.shape[:2] != recorte.shape[:2]:
            continue
        res = cv2.matchTemplate(recorte, template, cv2.TM_CCOEFF_NORMED)
        if np.max(res) >= threshold:
            return True
    return False

# Verifica se a imagem é só terreno estático
def eh_terreno(recorte, limite_variacao=8):
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    stddev = np.std(cinza)
    return stddev < limite_variacao

# Verifica se há vermelho ou amarelo no recorte
def contem_cor_desejada(recorte):
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

    mask_vermelho = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    mask_amarelo = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

    return cv2.countNonZero(mask_vermelho) > 15 or cv2.countNonZero(mask_amarelo) > 15

# Classificação manual com janela
def classificar_manual(recorte):
    temp_nome = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.png")
    cv2.imwrite(temp_nome, recorte)
    img = cv2.imread(temp_nome)

    janela_nome = "Novo objeto detectado"
    cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
    cv2.imshow(janela_nome, img)
    cv2.setWindowProperty(janela_nome, cv2.WND_PROP_TOPMOST, 1)

    print("\nNovo objeto encontrado. Classifique:")
    print("[m] Mob   [c] Coleta   [b] Mob Coleta   [i] Ignorar   [s] Sair")

    while True:
        tecla = cv2.waitKey(0) & 0xFF
        if tecla == ord('m'):
            pasta = MOBS_DIR
            break
        elif tecla == ord('c'):
            pasta = COLETAS_DIR
            break
        elif tecla == ord('b'):
            pasta = MOB_COLETA_DIR
            break
        elif tecla == ord('i'):
            pasta = IGNORADOS_DIR
            break
        elif tecla == ord('s'):
            print("Saindo...")
            cv2.destroyAllWindows()
            exit()

    nome_arquivo = f"{uuid.uuid4().hex}.png"
    destino = os.path.join(pasta, nome_arquivo)
    os.rename(temp_nome, destino)
    print(f"Salvo em: {destino}")
    cv2.destroyAllWindows()

# Loop principal com detecção de movimento
def main():
    print("Bot iniciado. Pressione [Ctrl+C] para sair.")
    altura, largura = 80, 80
    frame_anterior = None

    while True:
        frame_atual = capturar_tela_mss()
        if frame_atual is None:
            time.sleep(2)
            continue

        if frame_anterior is None:
            frame_anterior = frame_atual
            continue

        diff = cv2.absdiff(frame_anterior, frame_atual)
        frame_anterior = frame_atual

        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

        conhecidos = (
            carregar_templates(MOBS_DIR) +
            carregar_templates(COLETAS_DIR) +
            carregar_templates(MOB_COLETA_DIR) +
            carregar_templates(IGNORADOS_DIR)
        )

        for y in range(0, frame_atual.shape[0], altura):
            for x in range(0, frame_atual.shape[1], largura):
                recorte = frame_atual[y:y+altura, x:x+largura]
                recorte_diff = thresh[y:y+altura, x:x+largura]

                if recorte.shape[0] < altura or recorte.shape[1] < largura:
                    continue

                if cv2.countNonZero(recorte_diff) < 50:
                    continue

                if eh_terreno(recorte):
                    continue

                if not contem_cor_desejada(recorte):
                    continue

                if not ja_conhecido(recorte, conhecidos):
                    classificar_manual(recorte)
                    conhecidos = (
                        carregar_templates(MOBS_DIR) +
                        carregar_templates(COLETAS_DIR) +
                        carregar_templates(MOB_COLETA_DIR) +
                        carregar_templates(IGNORADOS_DIR)
                    )

        time.sleep(1)

if __name__ == "__main__":
    main()
