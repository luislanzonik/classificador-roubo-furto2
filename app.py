import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import warnings
import json

st.set_page_config(
    page_title="Classificador de Ocorrências",
    page_icon="📱",
    layout="centered"
)

st.title("Classificador de ocorrências de celulares")
st.caption("São Paulo • Base SSP-SP 2025 • Modelo MLP")

st.info(
    "Informe as características de uma ocorrência registrada. "
    "O modelo estima se o registro se parece mais com Furto ou Roubo, "
    "com base nos padrões aprendidos durante o treinamento."
)

st.warning(
    "Esta aplicação classifica registros informados pelo usuário. "
    "Ela não consulta ocorrências em tempo real, não conta crimes por bairro "
    "e não prevê onde ocorrerão crimes futuros."
)

MODEL_PATH = Path(__file__).resolve().parent / "modelo_mlp_roubo_furto.joblib"
MAPPING_PATH = Path(__file__).resolve().parent / "mapeamento_locais.json"

@st.cache_resource
def carregar_pacote(caminho):
    # O ambiente instala scikit-learn 1.6.1, mesma versão usada no treinamento.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pacote = joblib.load(caminho)
    if isinstance(pacote, dict) and "modelo" in pacote:
        return pacote
    return {"modelo": pacote}

def obter_categorias(modelo):
    """Extrai as categorias reais aprendidas pelo OneHotEncoder do pipeline."""
    categorias = {}
    try:
        etapas = getattr(modelo, "named_steps", {})
        for etapa in etapas.values():
            transformadores = getattr(etapa, "transformers_", [])
            for nome, transformador, colunas in transformadores:
                if isinstance(colunas, str):
                    colunas = [colunas]
                if hasattr(transformador, "categories_"):
                    for coluna, valores in zip(colunas, transformador.categories_):
                        categorias[str(coluna)] = sorted(
                            [str(v) for v in valores if pd.notna(v)],
                            key=lambda x: x.casefold()
                        )
                # Alguns pipelines podem ter um Pipeline dentro do ColumnTransformer.
                for subetapa in getattr(transformador, "steps", []):
                    sub = subetapa[1]
                    if hasattr(sub, "categories_"):
                        for coluna, valores in zip(colunas, sub.categories_):
                            categorias[str(coluna)] = sorted(
                                [str(v) for v in valores if pd.notna(v)],
                                key=lambda x: x.casefold()
                            )
    except Exception:
        pass
    return categorias

arquivo = st.file_uploader(
    "Se necessário, envie o modelo .joblib",
    type=["joblib", "pkl"],
    help="Normalmente não é necessário: o modelo já vem junto com esta aplicação."
)

pacote = None
try:
    if arquivo is not None:
        pacote = carregar_pacote(arquivo)
    elif MODEL_PATH.exists():
        pacote = carregar_pacote(str(MODEL_PATH))
except Exception as erro:
    st.error("Não foi possível abrir o modelo. Confira se a versão do scikit-learn é 1.6.1.")
    st.exception(erro)

if pacote is not None:
    modelo = pacote["modelo"]
    st.success("Modelo MLP carregado.")
    categorias = obter_categorias(modelo)

    colunas_categoricas = [
        ("BAIRRO", "Bairro"),
        ("DESCR_TIPOLOCAL", "Tipo de local"),
        ("DESCR_SUBTIPOLOCAL", "Subtipo do local"),
        ("DESCR_PERIODO", "Período do dia"),
    ]

    faltantes = [campo for campo, _ in colunas_categoricas if not categorias.get(campo)]
    if faltantes:
        st.error(
            "Não consegui recuperar as opções categóricas diretamente do modelo "
            f"({', '.join(faltantes)}). Para impedir entradas inventadas, a classificação "
            "foi desabilitada. É necessário exportar do notebook as categorias do treinamento."
        )
        st.stop()

    st.subheader("Dados da ocorrência")
    st.caption("As listas contêm categorias presentes no treinamento. O subtipo é filtrado conforme o tipo de local selecionado.")

    if not MAPPING_PATH.exists():
        st.error("O arquivo mapeamento_locais.json não foi encontrado. Adicione-o ao repositório junto com o app.py.")
        st.stop()

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapeamento_locais = json.load(f)

    # Mantém apenas tipos e subtipos também reconhecidos pelo modelo treinado.
    tipos_validos = [
        tipo for tipo in categorias["DESCR_TIPOLOCAL"]
        if tipo in mapeamento_locais
    ]
    if not tipos_validos:
        st.error("Não há tipos de local em comum entre o modelo e o mapeamento da base SSP-SP.")
        st.stop()

    bairro = st.selectbox("Bairro", categorias["BAIRRO"])
    tipo_local = st.selectbox("Tipo de local", tipos_validos)
    subtipos_do_tipo = [
        subtipo for subtipo in mapeamento_locais[tipo_local]
        if subtipo in categorias["DESCR_SUBTIPOLOCAL"]
    ]
    if not subtipos_do_tipo:
        st.error("Não há subtipos desse tipo disponíveis no modelo treinado.")
        st.stop()
    subtipo_local = st.selectbox("Subtipo do local", subtipos_do_tipo)
    periodo = st.selectbox("Período do dia", categorias["DESCR_PERIODO"])

    c1, c2, c3 = st.columns(3)
    with c1:
        mes = st.selectbox("Mês", list(range(1, 13)), format_func=lambda x: f"{x:02d}")
    with c2:
        dia_semana = st.selectbox(
            "Dia da semana",
            [
                ("Segunda-feira", 0), ("Terça-feira", 1), ("Quarta-feira", 2),
                ("Quinta-feira", 3), ("Sexta-feira", 4), ("Sábado", 5), ("Domingo", 6)
            ],
            format_func=lambda x: x[0]
        )
    with c3:
        hora = st.selectbox("Hora", list(range(24)), format_func=lambda x: f"{x:02d}:00")

    enviar = st.button("Classificar ocorrência", use_container_width=True)

    if enviar:
        entrada = pd.DataFrame([{
            "BAIRRO": bairro,
            "DESCR_TIPOLOCAL": tipo_local,
            "DESCR_SUBTIPOLOCAL": subtipo_local,
            "DESCR_PERIODO": periodo,
            "MES": int(mes),
            "DIA_SEMANA": int(dia_semana[1]),
            "HORA": int(hora)
        }])

        try:
            classe = modelo.predict(entrada)[0]
            probabilidades = modelo.predict_proba(entrada)[0]
            classes = list(modelo.classes_)

            # O treinamento do projeto usa 0 = Furto e 1 = Roubo.
            rotulos = {0: "Furto", 1: "Roubo"}
            if classe not in rotulos:
                st.error(f"O modelo retornou uma classe não reconhecida: {classe}")
                st.stop()

            prob_furto = float(probabilidades[classes.index(0)]) if 0 in classes else 0.0
            prob_roubo = float(probabilidades[classes.index(1)]) if 1 in classes else 0.0

            st.divider()
            st.subheader("Resultado")
            if classe == 1:
                st.error("Classificação estimada pelo modelo: ROUBO")
            else:
                st.success("Classificação estimada pelo modelo: FURTO")

            c1, c2 = st.columns(2)
            c1.metric("Estimativa para Furto", f"{prob_furto:.1%}")
            c2.metric("Estimativa para Roubo", f"{prob_roubo:.1%}")

            st.caption(
                "As porcentagens representam as estimativas do classificador para "
                "as características informadas; não são certeza nem avaliação de risco."
            )
        except Exception as erro:
            st.error(f"Não foi possível classificar esta entrada: {erro}")
else:
    st.error("O arquivo modelo_mlp_roubo_furto.joblib não foi encontrado na pasta da aplicação.")

st.divider()
st.caption("Projeto acadêmico • Classificação exploratória de registros de celulares subtraídos.")
