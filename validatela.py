import pyautogui
import pygetwindow as gw
import time
import cv2
import numpy as np
import os

print('Hellow worl')
NOME_JANELA = "Albion Online Client"
TEMPLATE_DIR = r"C:\Users\augusto\Documents\albionPy"
MOBS_DIR = os.path.join(TEMPLATE_DIR, "mobs")
COLETAS_DIR = os.path.join(TEMPLATE_DIR, "coletas")
MOB_COLETA_DIR = os.path.join(TEMPLATE_DIR, "mob_coleta")
IGNORADOS_DIR = os.path.join(TEMPLATE_DIR, "ignorados")
time.sleep(2)
# Tamanho do recorte para reconhecimento (exemplo 80x80)
TAM_RECORTE = 80

def pegar_janela():
    try:
        janela = gw.getWindowsWithTitle(NOME_JANELA)[0]
        if janela.isMinimized:
            print("Janela minimizada! Restaure para continuar.")
            return None
        janela.activate()
        time.sleep(0.3)
        return janela
    except IndexError:
        print(f"Janela '{NOME_JANELA}' não encontrada.")
        return None

def capturar_tela(janela):
    x, y, w, h = janela.left, janela.top, janela.width, janela.height
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame

def carregar_templates(pasta):
    templates = []
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        img = cv2.imread(caminho)
        if img is not None:
            templates.append(img)
    return templates

def match_templates(frame, templates, threshold=0.9):
    encontrados = []
    for template in templates:
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        h, w = template.shape[:2]
        for pt in zip(*loc[::-1]):
            encontrados.append((pt[0], pt[1], w, h))
    return encontrados

def reconhecer_alvos(frame):
    alvos = []

    mobs = carregar_templates(MOBS_DIR)
    coletas = carregar_templates(COLETAS_DIR)
    mob_coletas = carregar_templates(MOB_COLETA_DIR)

    encontrados_mobs = match_templates(frame, mobs, threshold=0.95)
    for (x, y, w, h) in encontrados_mobs:
        alvos.append({'tipo': 'mob', 'x': x, 'y': y, 'largura': w, 'altura': h})

    encontrados_mob_coleta = match_templates(frame, mob_coletas, threshold=0.95)
    for (x, y, w, h) in encontrados_mob_coleta:
        alvos.append({'tipo': 'mob_coleta', 'x': x, 'y': y, 'largura': w, 'altura': h})

    encontrados_coleta = match_templates(frame, coletas, threshold=0.95)
    for (x, y, w, h) in encontrados_coleta:
        alvos.append({'tipo': 'coleta', 'x': x, 'y': y, 'largura': w, 'altura': h})

    return alvos

def clicar_no_alvo(x, y, largura, altura, janela):
    pos_x = janela.left + x + largura // 2
    pos_y = janela.top + y + altura // 2
    pyautogui.moveTo(pos_x, pos_y, duration=0.2)
    pyautogui.click(button='right')
    return pos_x, pos_y

def coletar_recurso(x, y, largura, altura, janela):
    clicar_no_alvo(x, y, largura, altura, janela)
    time.sleep(0.5)
    time.sleep(3)  # tempo para coletar

def processar_alvos(alvos, janela):
    for alvo in alvos:
        if alvo['tipo'] != 'coleta':
            continue
        print("Indo coletar recurso...")
        coletar_recurso(alvo['x'], alvo['y'], alvo['largura'], alvo['altura'], janela)
        time.sleep(1)

def main():
    janela = pegar_janela()
    if janela is None:
        return

    print("Bot iniciado com reconhecimento automático! Pressione Ctrl+C para sair.")

    while True:
        frame = capturar_tela(janela)
        alvos_detectados = reconhecer_alvos(frame)
        processar_alvos(alvos_detectados, janela)
        time.sleep(2)

if __name__ == "__main__":
    main()
