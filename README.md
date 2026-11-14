# Classificador de ocorrências de roubo e furto de celulares — São Paulo (2025)

Projeto acadêmico que utiliza uma rede neural MLP (Multi-Layer Perceptron) para classificar registros de subtração de celulares nas categorias Roubo e Furto, com base em características registradas no boletim de ocorrência.

> **Atenção:** a aplicação classifica registros informados pelo usuário com base em padrões aprendidos no conjunto de treinamento. Não prevê crimes futuros, não estima risco de bairros e não deve ser usada para decisões policiais.

## Arquivos
- `app.py`: aplicação web em Streamlit.
- `modelo_mlp_roubo_furto.joblib`: pipeline MLP treinado e pré-processamento.
- `MLP_Roubo_Furto_SP_2025.ipynb`: notebook de tratamento, treinamento e avaliação.
- `requirements.txt`: dependências da aplicação.

## Executar localmente
Use Python compatível com as bibliotecas abaixo. Na pasta do projeto, execute:

```bash
pip install -r requirements.txt
streamlit run app.py
```

O arquivo `.joblib` deve ficar na mesma pasta do `app.py`. A versão do scikit-learn está fixada em `1.6.1`, correspondente à versão usada no treinamento do modelo.

## Dados
A base utilizada é `CelularesSubtraidos_2025.xlsx`, aba `CELULAR_2025`, disponibilizada pela SSP-SP. A planilha original não é incluída neste repositório por seu tamanho. Para executar novamente o notebook, obtenha a planilha na fonte oficial e disponibilize-a no Google Colab quando solicitado.

## Variáveis do modelo
- Bairro
- Tipo e subtipo do local
- Período do dia
- Mês
- Dia da semana
- Hora

## Limitações
A classificação depende da qualidade e representatividade dos registros usados no treinamento. As probabilidades são estimativas do modelo, não certezas nem avaliações de risco. O projeto tem finalidade acadêmica e exploratória.
