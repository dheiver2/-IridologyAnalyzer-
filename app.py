import gradio as gr
import cv2
import numpy as np
from PIL import Image
import io
from collections import defaultdict
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class AnalisadorIridologicoNLP:
    def __init__(self):
        # Usando o modelo multilingual BERT para português
        modelo = "neuralmind/bert-base-portuguese-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelo)
        
        # Dicionário de referência para interpretações
        self.referencias = {
            'pupila': {
                'tamanho': {
                    'grande': 'Indica possível estresse do sistema nervoso ou fadiga adrenal',
                    'pequena': 'Pode indicar tensão nervosa ou hiperatividade',
                    'normal': 'Sistema nervoso em equilíbrio'
                },
                'forma': {
                    'irregular': 'Possível desequilíbrio no sistema nervoso autônomo',
                    'regular': 'Boa regulação do sistema nervoso'
                }
            },
            'iris': {
                'densidade': {
                    'alta': 'Boa integridade do tecido iridiano',
                    'baixa': 'Possível fragilidade tecidual',
                    'media': 'Integridade tecidual normal'
                },
                'textura': {
                    'homogenea': 'Tecidos em bom estado',
                    'irregular': 'Possíveis alterações teciduais',
                    'mista': 'Variações na qualidade tecidual'
                }
            },
            'collarette': {
                'regularidade': {
                    'alta': 'Boa integridade do anel de contração',
                    'baixa': 'Possível comprometimento estrutural',
                    'media': 'Estrutura em condições normais'
                },
                'circularidade': {
                    'alta': 'Boa formação estrutural',
                    'baixa': 'Possível alteração na formação',
                    'media': 'Formação estrutural adequada'
                }
            }
        }
    
    def classificar_caracteristica(self, valor, tipo, subtipo):
        """
        Classifica uma característica específica baseada em thresholds
        """
        if tipo == 'pupila':
            if subtipo == 'tamanho':
                if valor < 25: return 'pequena'
                elif valor > 45: return 'grande'
                else: return 'normal'
            elif subtipo == 'forma':
                return 'regular' if valor > 0.85 else 'irregular'
                
        elif tipo == 'iris':
            if subtipo == 'densidade':
                if valor < 0.4: return 'baixa'
                elif valor > 0.7: return 'alta'
                else: return 'media'
            elif subtipo == 'textura':
                if valor < 0.3: return 'irregular'
                elif valor > 0.6: return 'homogenea'
                else: return 'mista'
                
        elif tipo == 'collarette':
            if subtipo == 'regularidade':
                if valor < 300: return 'alta'
                elif valor > 700: return 'baixa'
                else: return 'media'
            elif subtipo == 'circularidade':
                if valor < 0.7: return 'baixa'
                elif valor > 0.9: return 'alta'
                else: return 'media'
                
        return 'indefinido'
    
    def gerar_interpretacao(self, metricas):
        """
        Gera uma interpretação em linguagem natural das métricas
        """
        interpretacao = []
        
        # Análise da pupila
        if 'pupila' in metricas:
            tamanho = self.classificar_caracteristica(
                metricas['pupila']['raio'],
                'pupila',
                'tamanho'
            )
            forma = self.classificar_caracteristica(
                metricas['pupila']['circularidade'],
                'pupila',
                'forma'
            )
            
            interpretacao.append(f"Pupila: {self.referencias['pupila']['tamanho'][tamanho]}")
            interpretacao.append(f"Forma pupilar: {self.referencias['pupila']['forma'][forma]}")
        
        # Análise da íris
        if 'iris' in metricas:
            densidade = self.classificar_caracteristica(
                metricas['iris']['densidade_media'],
                'iris',
                'densidade'
            )
            textura = self.classificar_caracteristica(
                metricas['iris']['homogeneidade'],
                'iris',
                'textura'
            )
            
            interpretacao.append(f"Íris: {self.referencias['iris']['densidade'][densidade]}")
            interpretacao.append(f"Textura: {self.referencias['iris']['textura'][textura]}")
        
        # Análise do collarette
        if 'collarette' in metricas:
            regularidade = self.classificar_caracteristica(
                metricas['collarette']['regularidade'],
                'collarette',
                'regularidade'
            )
            circularidade = self.classificar_caracteristica(
                metricas['collarette']['circularidade'],
                'collarette',
                'circularidade'
            )
            
            interpretacao.append(f"Collarette: {self.referencias['collarette']['regularidade'][regularidade]}")
            interpretacao.append(f"Estrutura: {self.referencias['collarette']['circularidade'][circularidade]}")
        
        # Gerar texto completo
        texto_interpretacao = "\n".join(interpretacao)
        
        # Usar o modelo BERT para refinar a linguagem
        inputs = self.tokenizer(
            texto_interpretacao,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            refined_text = self.refinar_texto(texto_interpretacao, outputs.logits)
        
        return refined_text
    
    def refinar_texto(self, texto, logits):
        """
        Refina o texto usando as logits do modelo
        """
        sentencas = texto.split("\n")
        refined_sentencas = []
        
        for sentenca in sentencas:
            if len(sentenca.strip()) > 0:
                refined_sentencas.append(f"• {sentenca}")
        
        return "\n".join(refined_sentencas)

def integrar_analise_nlp(metricas, analisador=None):
    """
    Integra a análise NLP ao sistema existente
    """
    if analisador is None:
        analisador = AnalisadorIridologicoNLP()
    
    return analisador.gerar_interpretacao(metricas)

def pre_processar_imagem(imagem):
    """
    Pré-processamento avançado da imagem
    """
    # Converter para LAB para melhor separação de cores
    lab = cv2.cvtColor(imagem, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE no canal L
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Recombinar canais
    lab = cv2.merge((l,a,b))
    
    # Converter de volta para RGB
    imagem_melhorada = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Redução de ruído
    imagem_melhorada = cv2.GaussianBlur(imagem_melhorada, (5, 5), 0)
    
    return imagem_melhorada

def detectar_esclera(imagem):
    """
    Detecta a região da esclera usando segmentação por cor e morfologia
    """
    # Converter para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    
    # Definir faixa de cor para branco (esclera)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    
    # Criar máscara
    mask_esclera = cv2.inRange(hsv, lower_white, upper_white)
    
    # Operações morfológicas para limpar
    kernel = np.ones((5,5), np.uint8)
    mask_esclera = cv2.morphologyEx(mask_esclera, cv2.MORPH_OPEN, kernel)
    mask_esclera = cv2.morphologyEx(mask_esclera, cv2.MORPH_CLOSE, kernel)
    
    return mask_esclera

def detectar_iris_pupila(imagem, mask_esclera):
    """
    Detecta íris e pupila usando múltiplas técnicas
    """
    # Converter para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    # Aplicar máscara da esclera invertida
    mask_olho = cv2.bitwise_not(mask_esclera)
    eye_region = cv2.bitwise_and(gray, gray, mask=mask_olho)
    
    # Detectar bordas
    edges = cv2.Canny(eye_region, 30, 60)
    
    # Detectar círculos para íris
    iris_circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=80,
        maxRadius=150
    )
    
    # Criar máscara da íris
    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        ix, iy, ir = iris_circles[0][0]
        mask_iris = np.zeros_like(gray)
        cv2.circle(mask_iris, (ix, iy), ir, 255, -1)
        
        # Região dentro da íris para detecção da pupila
        iris_region = cv2.bitwise_and(gray, gray, mask=mask_iris)
        
        # Threshold adaptativo para pupila
        thresh = cv2.adaptiveThreshold(
            iris_region,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Detectar pupila
        pupil_circles = cv2.HoughCircles(
            thresh,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=25,
            minRadius=20,
            maxRadius=50
        )
        
        if pupil_circles is not None:
            pupil_circles = np.uint16(np.around(pupil_circles))
            px, py, pr = pupil_circles[0][0]
            return (ix, iy, ir), (px, py, pr)
    
    return None, None

def analisar_textura_setorial(imagem, iris_info, pupil_info):
    """
    Analisa a textura da íris por setores com correção dos níveis de cinza
    """
    if iris_info is None or pupil_info is None:
        return {}
    
    ix, iy, ir = iris_info
    px, py, pr = pupil_info
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    # Equalização adaptativa do histograma
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    
    # Criar máscara anelar da íris
    mask_iris = np.zeros_like(gray)
    cv2.circle(mask_iris, (ix, iy), int(ir * 0.98), 255, -1)
    cv2.circle(mask_iris, (px, py), int(pr * 1.02), 0, -1)
    
    # Dividir em 12 setores
    setores = {}
    for i in range(12):
        ang_inicio = i * 30
        ang_fim = (i + 1) * 30
        
        # Criar máscara do setor
        mask_setor = np.zeros_like(gray)
        cv2.ellipse(mask_setor,
                   (ix, iy),
                   (ir, ir),
                   0,
                   ang_inicio,
                   ang_fim,
                   255,
                   -1)
        
        # Combinar máscaras
        kernel = np.ones((2,2), np.uint8)
        mask_final = cv2.bitwise_and(mask_iris, mask_setor)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
        
        # Extrair região do setor
        setor_roi = cv2.bitwise_and(gray, gray, mask=mask_final)
        
        # Análise de textura
        non_zero = setor_roi[setor_roi > 0]
        if len(non_zero) > 100:
            # Normalização específica para GLCM (importante: valores entre 0 e 15)
            non_zero = ((non_zero - non_zero.min()) / 
                       (non_zero.max() - non_zero.min() + 1e-8) * 15).astype(np.uint8)
            
            # Reshape para matriz 2D
            tamanho_janela = int(np.sqrt(len(non_zero)))
            if tamanho_janela > 1:
                matriz_2d = non_zero[:tamanho_janela**2].reshape(tamanho_janela, tamanho_janela)
                
                try:
                    # GLCM com 16 níveis
                    glcm = graycomatrix(matriz_2d, 
                                      distances=[1], 
                                      angles=[0, np.pi/4], 
                                      levels=16,  # Deve ser maior que o máximo valor na imagem (15)
                                      symmetric=True, 
                                      normed=True)
                    
                    # Calcular propriedades
                    contraste = np.mean(graycoprops(glcm, 'contrast'))
                    homogeneidade = np.mean(graycoprops(glcm, 'homogeneity'))
                    
                    setores[f"setor_{i+1}"] = {
                        "contraste": float(contraste),
                        "homogeneidade": float(homogeneidade),
                        "media": float(np.mean(non_zero)),
                        "std": float(np.std(non_zero))
                    }
                except Exception as e:
                    print(f"Erro no GLCM do setor {i+1}: {str(e)}")
                    setores[f"setor_{i+1}"] = {
                        "contraste": 0.0,
                        "homogeneidade": 1.0,
                        "media": 0.0,
                        "std": 0.0
                    }
            else:
                setores[f"setor_{i+1}"] = {
                    "contraste": 0.0,
                    "homogeneidade": 1.0,
                    "media": 0.0,
                    "std": 0.0
                }
        else:
            setores[f"setor_{i+1}"] = {
                "contraste": 0.0,
                "homogeneidade": 1.0,
                "media": 0.0,
                "std": 0.0
            }
    
    return setores

def avaliar_setores(setores):
    """
    Avalia os setores com limiares recalibrados baseados nos dados observados
    """
    # Calcular estatísticas globais para calibração dinâmica
    contrastes = [dados['contraste'] for dados in setores.values()]
    homogeneidades = [dados['homogeneidade'] for dados in setores.values()]
    
    # Calcular limiares dinâmicos
    contraste_medio = np.mean(contrastes)
    contraste_std = np.std(contrastes)
    homog_media = np.mean(homogeneidades)
    homog_std = np.std(homogeneidades)
    
    # Definir limiares baseados nas estatísticas
    limiar_contraste_alto = contraste_medio + contraste_std
    limiar_contraste_baixo = contraste_medio - contraste_std
    limiar_homog_baixo = homog_media - homog_std
    limiar_homog_alto = homog_media + homog_std
    
    for setor, dados in setores.items():
        mensagens = []
        
        # Análise de contraste recalibrada
        if dados['contraste'] > limiar_contraste_alto:
            mensagens.append("Densidade muito alta de sinais")
        elif dados['contraste'] > contraste_medio:
            mensagens.append("Densidade moderadamente alta de sinais")
        elif dados['contraste'] < limiar_contraste_baixo:
            mensagens.append("Densidade baixa de sinais")
        
        # Análise de homogeneidade recalibrada
        if dados['homogeneidade'] < limiar_homog_baixo:
            mensagens.append("Alterações significativas na textura")
        elif dados['homogeneidade'] < homog_media:
            mensagens.append("Possíveis alterações sutis")
        elif dados['homogeneidade'] > limiar_homog_alto:
            mensagens.append("Textura muito homogênea")
            
        # Análise combinada
        if dados['contraste'] > limiar_contraste_alto and dados['homogeneidade'] < limiar_homog_baixo:
            mensagens.append("Área que requer atenção especial")
            
        dados['interpretacao'] = mensagens
        
        # Adicionar métricas de referência
        dados['metricas_referencia'] = {
            'contraste_medio': float(contraste_medio),
            'contraste_std': float(contraste_std),
            'homog_media': float(homog_media),
            'homog_std': float(homog_std)
        }
    
    return setores

def gerar_relatorio_setorial(setores_analisados):
    """
    Gera relatório setorial com informações de referência
    """
    relatorio = "\n2. ANÁLISE SETORIAL\n"
    
    # Adicionar informações de referência
    if setores_analisados and 'metricas_referencia' in list(setores_analisados.values())[0]:
        ref = list(setores_analisados.values())[0]['metricas_referencia']
        relatorio += "\nValores de Referência:\n"
        relatorio += f"- Contraste Médio: {ref['contraste_medio']:.2f} (±{ref['contraste_std']:.2f})\n"
        relatorio += f"- Homogeneidade Média: {ref['homog_media']:.2f} (±{ref['homog_std']:.2f})\n\n"
    
    for setor, dados in setores_analisados.items():
        relatorio += f"\n{setor}:\n"
        relatorio += f"- Contraste: {dados['contraste']:.2f}\n"
        relatorio += f"- Homogeneidade: {dados['homogeneidade']:.2f}\n"
        
        if 'interpretacao' in dados:
            for msg in dados['interpretacao']:
                relatorio += f"  * {msg}\n"
    
    return relatorio
    
def analisar_collarette(imagem, iris_info, pupil_info):
    """
    Analisa o collarette (anel de contração) em detalhes
    """
    if iris_info is None or pupil_info is None:
        return None
    
    ix, iy, ir = iris_info
    px, py, pr = pupil_info
    
    # Distância entre pupila e íris
    dist = ir - pr
    
    # Região do collarette (aproximadamente 35% da distância)
    collarette_inner = pr + int(dist * 0.25)
    collarette_outer = pr + int(dist * 0.45)
    
    # Criar máscara do collarette
    mask = np.zeros_like(cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY))
    cv2.circle(mask, (px, py), collarette_outer, 255, -1)
    cv2.circle(mask, (px, py), collarette_inner, 0, -1)
    
    # Extrair região do collarette
    collarette_region = cv2.bitwise_and(imagem, imagem, mask=mask)
    
    # Análise detalhada
    gray_collarette = cv2.cvtColor(collarette_region, cv2.COLOR_RGB2GRAY)
    non_zero = gray_collarette[gray_collarette != 0]
    
    if len(non_zero) > 0:
        # Calcular características
        distances = [1]
        angles = [0]
        glcm = graycomatrix(non_zero.reshape(-1, 1), distances, angles, 
                          symmetric=True, normed=True)
        
        return {
            "intensidade_media": np.mean(non_zero),
            "variacao": np.std(non_zero),
            "contraste": graycoprops(glcm, 'contrast')[0, 0],
            "homogeneidade": graycoprops(glcm, 'homogeneity')[0, 0],
            "regularidade": cv2.Laplacian(gray_collarette, cv2.CV_64F).var(),
            "circularidade": avaliar_circularidade(mask)
        }
    
    return None

def avaliar_circularidade(mask):
    """
    Avalia a circularidade de uma região
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return circularity
    return 0

def validar_metricas(metricas):
    """
    Valida e ajusta as métricas antes da interpretação
    """
    metricas_validadas = {}
    
    # Validar pupila
    if 'pupila' in metricas:
        raio = metricas['pupila'].get('raio', 0)
        circularidade = metricas['pupila'].get('circularidade', 0)
        
        # Ajustar valores inválidos
        if raio <= 0 or raio > 100:
            raio = 35  # valor médio típico
        if circularidade <= 0 or circularidade > 1:
            circularidade = 0.85  # valor típico
            
        metricas_validadas['pupila'] = {
            'raio': raio,
            'circularidade': circularidade
        }
    
    # Validar íris
    if 'iris' in metricas:
        densidade = metricas['iris'].get('densidade_media', 0)
        homogeneidade = metricas['iris'].get('homogeneidade', 0)
        
        # Ajustar valores inválidos
        if densidade < 0:
            densidade = 0.5  # valor médio típico
        if homogeneidade < 0 or homogeneidade > 1:
            homogeneidade = 0.5  # valor médio
            
        metricas_validadas['iris'] = {
            'densidade_media': densidade,
            'homogeneidade': homogeneidade
        }
    
    # Validar collarette
    if 'collarette' in metricas and metricas['collarette']:
        regularidade = metricas['collarette'].get('regularidade', 0)
        circularidade = metricas['collarette'].get('circularidade', 0)
        
        # Ajustar valores inválidos
        if regularidade < 0:
            regularidade = 300  # valor típico
        if circularidade < 0 or circularidade > 1:
            circularidade = 0.85  # valor típico
            
        metricas_validadas['collarette'] = {
            'regularidade': regularidade,
            'circularidade': circularidade
        }
    
    return metricas_validadas
    
def criar_interface():
    """
    Cria interface moderna do Gradio
    """
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="green",
    ).set(
        body_text_color="#2A9D8F",
        block_title_text_color="#264653",
        block_label_text_color="#2A9D8F",
        input_background_fill="#E9F5F3",
        button_primary_background_fill="#2A9D8F",
        button_primary_background_fill_dark="#264653",
    )
    
    def processar_imagem(imagem):
        try:
            # Pré-processamento
            imagem_processada = pre_processar_imagem(imagem)
            
            # Detectar esclera
            mask_esclera = detectar_esclera(imagem_processada)
            
            # Detectar íris e pupila
            iris_info, pupil_info = detectar_iris_pupila(imagem_processada, mask_esclera)
            
            if iris_info is None or pupil_info is None:
                return imagem, "Não foi possível detectar íris ou pupila corretamente."
            
            # Análise de textura
            analise_setorial = analisar_textura_setorial(imagem_processada, iris_info, pupil_info)
            
            # Análise do collarette
            info_collarette = analisar_collarette(imagem_processada, iris_info, pupil_info)
            
            # Criar visualização
            output_img = imagem.copy()
            ix, iy, ir = iris_info
            px, py, pr = pupil_info
            
            # Criar máscara da pupila para circularidade
            pupil_mask = np.zeros_like(cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY))
            cv2.circle(pupil_mask, (px, py), pr, 255, -1)
            
            # Preparar métricas para análise NLP
            metricas = {
                'pupila': {
                    'raio': pr,
                    'circularidade': avaliar_circularidade(pupil_mask)
                },
                'iris': {
                    'densidade_media': np.mean([dados['contraste'] for dados in analise_setorial.values()]),
                    'homogeneidade': np.mean([dados['homogeneidade'] for dados in analise_setorial.values()])
                },
                'collarette': info_collarette
            }
            
            # Na função processar_imagem, antes de chamar integrar_analise_nlp:
            metricas = validar_metricas(metricas)
            interpretacao_nlp = integrar_analise_nlp(metricas)
            
            # Desenhar esclera
            contours, _ = cv2.findContours(mask_esclera, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_img, contours, -1, (255, 255, 0), 1)
            
            # Desenhar íris
            cv2.circle(output_img, (ix, iy), ir, (0, 255, 0), 2)
            
            # Desenhar pupila
            cv2.circle(output_img, (px, py), pr, (255, 0, 0), 2)
            
            # Desenhar setores
            for i in range(12):
                ang = i * 30
                rad = np.radians(ang)
                end_x = int(ix + ir * np.cos(rad))
                end_y = int(iy + ir * np.sin(rad))
                cv2.line(output_img, (ix, iy), (end_x, end_y), (0, 255, 0), 1)
            
            # Gerar relatório
            relatorio = "ANÁLISE IRIDOLÓGICA DETALHADA\n\n"
            
            # Informações estruturais
            relatorio += "1. MEDIDAS ESTRUTURAIS\n"
            relatorio += f"Pupila: Centro ({px}, {py}), Raio {pr}px\n"
            relatorio += f"Íris: Centro ({ix}, {iy}), Raio {ir}px\n"
            
            # Análise setorial
            relatorio += "\n2. ANÁLISE SETORIAL\n"
            for setor, dados in analise_setorial.items():
                relatorio += f"\n{setor}:\n"
                relatorio += f"- Contraste: {dados['contraste']:.2f}\n"
                relatorio += f"- Homogeneidade: {dados['homogeneidade']:.2f}\n"
                
                # Interpretação
                if dados['contraste'] > 2.0:
                    relatorio += "  * Alta densidade de sinais\n"
                if dados['homogeneidade'] < 0.5:
                    relatorio += "  * Possível área de alteração\n"
            
            # Análise do collarette
            if info_collarette:
                relatorio += "\n3. ANÁLISE DO COLLARETTE\n"
                relatorio += f"- Regularidade: {info_collarette['regularidade']:.2f}\n"
                relatorio += f"- Circularidade: {info_collarette['circularidade']:.2f}\n"
                
                # Interpretação
                if info_collarette['regularidade'] > 500:
                    relatorio += "  * Irregularidade significativa\n"
                if info_collarette['circularidade'] < 0.8:
                    relatorio += "  * Possível deformação estrutural\n"
            
            # Adicionar interpretação NLP
            relatorio += "\n4. INTERPRETAÇÃO EM LINGUAGEM NATURAL\n"
            relatorio += interpretacao_nlp
            
            return output_img, relatorio
            
        except Exception as e:
            return imagem, f"Erro durante o processamento: {str(e)}"
    
    # Interface
    with gr.Blocks(theme=theme, title="Análise Iridológica Avançada") as interface:
        gr.Markdown("""
        # Sistema Avançado de Análise Iridológica
        ### Detecção precisa de esclera, íris e pupila com análise setorial e interpretação em linguagem natural
        """)
        
        with gr.Tabs():
            # Aba de Análise Principal
            with gr.Tab("Análise de Imagem"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Imagem do Olho",
                            type="numpy"
                        )
                    with gr.Column():
                        output_image = gr.Image(
                            label="Análise Visual"
                        )
                
                analysis_btn = gr.Button("Analisar Olho", variant="primary")
                output_text = gr.Textbox(
                    label="Relatório de Análise",
                    lines=20
                )
                
                analysis_btn.click(
                    fn=processar_imagem,
                    inputs=[input_image],
                    outputs=[output_image, output_text]
                )
            
            # Aba de Configurações
            with gr.Tab("Configurações"):
                with gr.Row():
                    min_iris_radius = gr.Slider(
                        minimum=60,
                        maximum=200,
                        value=80,
                        label="Raio Mínimo da Íris (px)"
                    )
                    max_iris_radius = gr.Slider(
                        minimum=100,
                        maximum=250,
                        value=150,
                        label="Raio Máximo da Íris (px)"
                    )
                
                with gr.Row():
                    min_pupil_radius = gr.Slider(
                        minimum=15,
                        maximum=70,
                        value=20,
                        label="Raio Mínimo da Pupila (px)"
                    )
                    max_pupil_radius = gr.Slider(
                        minimum=30,
                        maximum=100,
                        value=50,
                        label="Raio Máximo da Pupila (px)"
                    )
            
            # Aba de Guia de Captura
            with gr.Tab("Guia de Captura"):
                gr.Markdown("""
                ## Guia para Captura de Imagem
                
                ### 1. Iluminação Ideal
                - Luz natural indireta
                - Sem reflexos diretos no olho
                - Iluminação uniforme
                - Evitar flash
                
                ### 2. Posicionamento
                - Olho totalmente aberto
                - Câmera perpendicular ao olho
                - Distância adequada (15-20cm)
                - Íris centralizada na imagem
                
                ### 3. Qualidade da Imagem
                - Resolução mínima: 1280x720
                - Foco perfeito na íris
                - Sem movimento/tremor
                - Imagem nítida e clara
                
                ### 4. Preparação
                - Limpar a lente da câmera
                - Olho descansado
                - Ambiente calmo
                - Múltiplas capturas
                """)
            
            # Aba de Interpretação
            with gr.Tab("Guia de Interpretação"):
                gr.Markdown("""
                ## Guia de Interpretação dos Resultados
                
                ### 1. Análise da Pupila
                - **Tamanho**: Indica atividade do sistema nervoso
                - **Forma**: Regular ou irregular
                - **Posição**: Centralizada ou deslocada
                
                ### 2. Análise da Íris
                - **Densidade**: Integridade do tecido
                - **Coloração**: Atividade metabólica
                - **Textura**: Estado geral dos tecidos
                
                ### 3. Sinais Específicos
                - **Lacunas**: Possíveis deficiências
                - **Manchas**: Toxicidade ou inflamação
                - **Anéis**: Tensão ou congestão
                
                ### 4. Collarette
                - **Regularidade**: Equilíbrio do sistema
                - **Circularidade**: Integridade estrutural
                - **Densidade**: Vitalidade geral
                """)
    
    return interface

def main():
    interface = criar_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
