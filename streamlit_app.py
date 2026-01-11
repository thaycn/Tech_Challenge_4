import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# -----------------------------
# Configurações da página
# -----------------------------
st.set_page_config(
    page_title="Preditor de Obesidade - Tech Challenge",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Preditor de Nível de Obesidade")
st.write(
    "Aplicação preditiva para apoiar a equipe médica na avaliação do nível de obesidade, "
    "com base em dados demográficos e comportamentais."
)

# -----------------------------
# Carregamento do modelo
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/pipeline_obesity_model.joblib")
    if not model_path.exists():
        st.error("Modelo não encontrado. Verifique se o arquivo está em models/pipeline_obesity_model.joblib")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Opções (categorias) conforme dataset
# -----------------------------
GENDER_OPTIONS = ["Female", "Male"]
YES_NO = ["no", "yes"]
FREQ_OPTIONS = ["no", "Sometimes", "Frequently", "Always"]
MTRANS_OPTIONS = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]

def interpret_prediction(top_class: str, top_prob: float, second_class: str, second_prob: float) -> str:
    """
    Gera uma interpretação simples e segura baseada nas probabilidades.
    Regras pensadas para apoiar a decisão (não são diagnóstico).
    """
    top_pct = top_prob * 100
    second_pct = second_prob * 100
    gap = top_pct - second_pct

    if top_pct >= 70 and gap >= 15:
        return (
            f"Alta confiança: o modelo indica **{top_class}** com **{top_pct:.1f}%**. "
            f"A segunda hipótese (**{second_class}**) ficou bem abaixo (**{second_pct:.1f}%**)."
        )
    if top_pct >= 50 and gap >= 10:
        return (
            f"Confiança moderada: o modelo indica **{top_class}** com **{top_pct:.1f}%**, "
            f"com **{second_class}** como segunda hipótese (**{second_pct:.1f}%**)."
        )
    if gap < 10:
        return (
            f"Caso limítrofe: as classes **{top_class}** (**{top_pct:.1f}%**) e **{second_class}** "
            f"(**{second_pct:.1f}%**) estão próximas. Recomenda-se avaliar clinicamente e, se possível, "
            "coletar informações adicionais."
        )
    return (
        f"Baixa confiança: a maior probabilidade foi **{top_class}** (**{top_pct:.1f}%**), "
        "mas o modelo está relativamente incerto. Recomenda-se cautela na interpretação."
    )

st.subheader("🧾 Dados do paciente")

with st.form("patient_form"):
    st.markdown("### Informações demográficas e antropométricas")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender (Gênero)", GENDER_OPTIONS)

        age = st.number_input("Age (Idade)", min_value=0.0, max_value=120.0, value=24.0, step=1.0)
        st.caption("Idade em anos.")

        height = st.number_input("Height (Altura em metros)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
        st.caption("Altura em metros (ex.: 1.70).")

    with col2:
        weight = st.number_input("Weight (Peso em kg)", min_value=10.0, max_value=300.0, value=86.0, step=0.5)
        st.caption("Peso em quilogramas (kg).")

        family_history = st.selectbox("family_history (Histórico familiar de sobrepeso?)", YES_NO)
        smoke = st.selectbox("SMOKE (Fuma?)", YES_NO)

    st.markdown("### Hábitos alimentares e estilo de vida")

    col3, col4 = st.columns(2)
    with col3:
        favc = st.selectbox("FAVC (Alimentos altamente calóricos com frequência?)", YES_NO)
        caec = st.selectbox("CAEC (Come entre refeições?)", FREQ_OPTIONS)
        calc = st.selectbox("CALC (Frequência de consumo de álcool)", FREQ_OPTIONS)
        scc = st.selectbox("SCC (Monitora calorias diariamente?)", YES_NO)

    with col4:
        mtrans = st.selectbox("MTRANS (Meio de transporte)", MTRANS_OPTIONS)

        st.caption("Obs.: algumas variáveis numéricas podem assumir valores decimais (ex.: 1.7), pois vêm de escalas contínuas no dataset.")

        fcvc = st.slider("FCVC (Consumo de vegetais)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        st.caption("1 = baixo consumo | 2 = moderado | 3 = alto")

        ncp = st.slider("NCP (Número de refeições principais)", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        st.caption("Quantidade de refeições principais por dia.")

        ch2o = st.slider("CH2O (Consumo de água)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        st.caption("1 = baixa ingestão | 2 = moderada | 3 = alta")

    st.markdown("### Atividade física e tempo de tela")
    col5, col6 = st.columns(2)
    with col5:
        faf = st.slider("FAF (Frequência de atividade física)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        st.caption("0 = nenhuma | 1 = baixa | 2 = moderada | 3 = alta")

    with col6:
        tue = st.slider("TUE (Tempo de uso de tecnologia)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        st.caption("0 = baixo | 1 = moderado | 2 = alto (equivalente a TER no enunciado)")

    submitted = st.form_submit_button("🔎 Predizer nível de obesidade")

# -----------------------------
# Predição
# -----------------------------
if submitted:
    bmi = weight / (height ** 2)
    st.info(f"🧮 BMI (IMC) calculado automaticamente: **{bmi:.2f}**")

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": float(age),
        "Height": float(height),
        "Weight": float(weight),
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": float(fcvc),
        "NCP": float(ncp),
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": float(ch2o),
        "SCC": scc,
        "FAF": float(faf),
        "TUE": float(tue),
        "CALC": calc,
        "MTRANS": mtrans,
        "BMI": float(bmi),
    }])

    pred = model.predict(input_data)[0]

    # Probabilidades
    top_prob = None
    proba_df = None
    interpretation = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        classes = model.classes_

        proba_df = pd.DataFrame({
            "Classe": classes,
            "Probabilidade": proba
        }).sort_values("Probabilidade", ascending=False)

        # Top-1 e Top-2
        top_class = proba_df.iloc[0]["Classe"]
        top_prob = float(proba_df.iloc[0]["Probabilidade"])
        second_class = proba_df.iloc[1]["Classe"]
        second_prob = float(proba_df.iloc[1]["Probabilidade"])

        interpretation = interpret_prediction(top_class, top_prob, second_class, second_prob)

        st.success(f"✅ Predição do modelo: **{pred}** — **{top_prob*100:.1f}%**")

        st.markdown("### 🩺 Interpretação")
        st.write(interpretation)

        st.markdown("### 📊 Probabilidades por classe")
        proba_df_display = proba_df.copy()
        proba_df_display["Probabilidade"] = (proba_df_display["Probabilidade"] * 100).map(lambda x: f"{x:.1f}%")
        st.dataframe(proba_df_display, use_container_width=True)

    else:
        st.success(f"✅ Predição do modelo: **{pred}**")

    st.markdown("---")
    st.markdown("### ℹ️ Observação")
    st.write(
        "Este sistema é um apoio à decisão e não substitui avaliação clínica. "
        "Os resultados devem ser interpretados por profissionais de saúde."
    )
