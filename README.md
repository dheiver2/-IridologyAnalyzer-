# Sistema Avançado de Análise Iridológica

Um sistema completo de análise iridológica utilizando visão computacional e processamento de linguagem natural para análise e interpretação de imagens da íris.

## Descrição

Este projeto implementa um sistema avançado de análise iridológica que combina técnicas de processamento de imagem, visão computacional e processamento de linguagem natural para realizar análises detalhadas da íris. O sistema é capaz de detectar e analisar diferentes aspectos da íris, incluindo pupila, collarette e setores específicos, gerando relatórios detalhados com interpretações em linguagem natural.

## Funcionalidades Principais

- Detecção automática de esclera, íris e pupila
- Análise setorial da íris (12 setores)
- Análise do collarette (anel de contração)
- Processamento avançado de imagem
- Interpretação em linguagem natural usando BERT
- Interface gráfica intuitiva

## Requisitos do Sistema

### Dependências Principais
```
gradio
opencv-python
numpy
Pillow
scipy
scikit-image
torch
transformers
```

### Modelo de Linguagem
- BERT multilingual para português (neuralmind/bert-base-portuguese-cased)

## Instalação

1. Clone o repositório:
```bash
git clone [url-do-repositorio]
cd analise-iridologica
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

1. Execute o programa principal:
```bash
python main.py
```

2. Acesse a interface web através do navegador no endereço fornecido.

3. Faça upload de uma imagem do olho seguindo as diretrizes de captura.

## Funcionalidades Detalhadas

### Processamento de Imagem
- Pré-processamento com correção de cor e redução de ruído
- Detecção de bordas e segmentação
- Análise de textura usando GLCM
- Avaliação de circularidade e regularidade

### Análise Setorial
- Divisão da íris em 12 setores
- Análise individual de cada setor
- Métricas de contraste e homogeneidade
- Detecção de padrões e anomalias

### Interpretação
- Análise da pupila (tamanho e forma)
- Avaliação da densidade da íris
- Análise do collarette
- Geração de relatórios detalhados

## Interface do Usuário

### Abas Principais
1. **Análise de Imagem**
   - Upload de imagem
   - Visualização da análise
   - Relatório detalhado

2. **Configurações**
   - Ajuste de parâmetros
   - Calibração do sistema

3. **Guia de Captura**
   - Instruções para fotografia
   - Requisitos técnicos
   - Boas práticas

4. **Guia de Interpretação**
   - Explicação dos resultados
   - Significado das métricas
   - Referências de análise

## Guia de Captura de Imagem

### Iluminação
- Luz natural indireta
- Evitar reflexos
- Iluminação uniforme
- Sem flash

### Posicionamento
- Olho completamente aberto
- Câmera perpendicular
- Distância: 15-20cm
- Íris centralizada

### Qualidade
- Resolução mínima: 1280x720
- Foco preciso
- Imagem estável
- Alta nitidez

## Considerações Técnicas

### Processamento de Imagem
- Conversão para espaço de cor LAB
- Equalização adaptativa de histograma
- Detecção de círculos usando transformada de Hough
- Análise de textura GLCM

### Análise NLP
- Modelo BERT multilingual
- Tokenização e processamento de texto
- Geração de interpretações naturais
- Refinamento de linguagem

## Contribuição

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença [inserir tipo de licença]. Veja o arquivo `LICENSE` para mais detalhes.

## Contato

[Inserir informações de contato]

## Reconhecimentos

- OpenCV pela biblioteca de visão computacional
- Hugging Face pela implementação do BERT
- Gradio pela framework de interface
