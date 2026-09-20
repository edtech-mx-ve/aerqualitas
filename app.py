"""AerQualitas Streamlit entry point — consolidated user interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aerqualitas.config import SETTINGS
from aerqualitas.inference.engine import (
    load_inference_assets,
    load_inference_metadata,
    predict_pm25,
)
from aerqualitas.inference.validation import InputValidationError
from aerqualitas.logging_config import configure_logging
from aerqualitas.paths import ASSETS_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from aerqualitas.validation import validate_required_file


@st.cache_data(show_spinner=False)
def load_csv(path: str, *, parse_datetime: bool = False) -> pd.DataFrame:
    """Load an application CSV artifact."""
    kwargs = {"parse_dates": ["datetime"]} if parse_datetime else {}
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, object]:
    """Load a JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner="Cargando modelo validado...")
def load_final_inference_assets(
    model_path: str,
    preprocessor_path: str,
    metadata_path: str,
    manifest_path: str,
):
    """Load and cache final inference artifacts after integrity verification."""
    return load_inference_assets(
        model_path=Path(model_path),
        preprocessor_path=Path(preprocessor_path),
        metadata_path=Path(metadata_path),
        final_manifest_path=Path(manifest_path),
    )


def apply_visual_style() -> None:
    """Apply the final AerQualitas visual identity without changing model logic."""
    st.markdown(
        """
        <style>
        :root {
            --aq-primary: #2F7FA8;
            --aq-primary-dark: #1F5F82;
            --aq-sky: #83CEF4;
            --aq-ink: #17324D;
            --aq-muted: #617887;
            --aq-card: #F7FBFD;
            --aq-border: #DCECF4;
            --aq-soft: #EEF7FB;
        }

        .stApp {
            background: linear-gradient(180deg, #FFFFFF 0%, #F9FCFE 100%);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F3F9FC 0%, #EDF6FA 100%);
            border-right: 1px solid var(--aq-border);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.32rem 0.2rem;
            border-radius: 9px;
        }

        h1, h2, h3, h4 {
            color: var(--aq-ink);
        }

        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--aq-border);
            border-radius: 16px;
            padding: 0.75rem 0.95rem;
            box-shadow: 0 3px 14px rgba(34, 96, 128, 0.06);
        }

        div[data-testid="stMetric"] label {
            color: #456678;
        }

        .aq-hero {
            background:
                radial-gradient(circle at 85% 10%, rgba(131,206,244,.30), transparent 30%),
                linear-gradient(135deg, #F7FCFF 0%, #EEF8FD 100%);
            border: 1px solid var(--aq-border);
            border-radius: 22px;
            padding: 1.2rem 1.35rem;
            margin: 0.2rem 0 1.1rem 0;
        }

        .aq-eyebrow {
            color: var(--aq-primary-dark);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
            margin-bottom: .3rem;
        }

        .aq-hero-title {
            color: var(--aq-ink);
            font-size: 1.55rem;
            line-height: 1.25;
            font-weight: 750;
            margin-bottom: .35rem;
        }

        .aq-hero-copy {
            color: var(--aq-muted);
            margin: 0;
            max-width: 900px;
        }

        .aq-card {
            background: #FFFFFF;
            border: 1px solid var(--aq-border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 220px;
            box-shadow: 0 4px 18px rgba(34, 96, 128, 0.06);
        }

        .aq-card-title {
            color: var(--aq-primary-dark);
            font-size: 1.03rem;
            font-weight: 750;
            margin-bottom: .6rem;
        }

        .aq-card-value {
            color: var(--aq-ink);
            font-size: 1.55rem;
            font-weight: 760;
            line-height: 1.1;
            margin-bottom: .35rem;
        }

        .aq-card-copy {
            color: var(--aq-muted);
            font-size: .92rem;
            line-height: 1.45;
        }

        .aq-pill {
            display: inline-block;
            background: var(--aq-soft);
            border: 1px solid var(--aq-border);
            color: var(--aq-primary-dark);
            border-radius: 999px;
            padding: .25rem .55rem;
            margin: .12rem .15rem .12rem 0;
            font-size: .78rem;
            font-weight: 650;
        }

        .aq-architecture {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: .35rem;
            margin-top: .8rem;
        }

        .aq-node {
            background: #F5FBFE;
            border: 1px solid #CFE5F0;
            color: #214D66;
            padding: .35rem .55rem;
            border-radius: 10px;
            font-size: .82rem;
            font-weight: 650;
        }

        .aq-arrow {
            color: #6D9DB7;
            font-weight: 700;
        }

        .aq-flow-step {
            display: flex;
            gap: .6rem;
            align-items: flex-start;
            margin: .5rem 0;
            color: #39596A;
            font-size: .9rem;
        }

        .aq-step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.45rem;
            height: 1.45rem;
            flex: 0 0 1.45rem;
            border-radius: 50%;
            background: #DFF1FA;
            color: #1F678D;
            font-weight: 750;
            font-size: .8rem;
        }

        .aq-section-kicker {
            color: var(--aq-primary-dark);
            font-size: .84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .035em;
            margin-top: .25rem;
        }

        .aq-footer {
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #E3EDF2;
            color: #667985;
            font-size: 0.86rem;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2F7FA8 0%, #3A93BE 100%);
            border: 0;
            color: #FFFFFF;
            font-weight: 700;
            border-radius: 10px;
            min-height: 2.65rem;
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, #276D91 0%, #337FA4 100%);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: .8rem;
            }
            .aq-card {
                min-height: auto;
            }
            .aq-hero-title {
                font-size: 1.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render AerQualitas identity with the approved logo in the main header."""
    logo_path = ASSETS_DIR / "logo.png"
    left, right = st.columns([1.1, 8.9], vertical_alignment="center")

    with left:
        if logo_path.exists():
            st.image(str(logo_path), width=74)

    with right:
        st.title(SETTINGS.name)
        st.caption(
            "Evaluación de la calidad del aire mediante estimación de PM2.5 · "
            "Datos, inteligencia artificial y ciencia para un aire más saludable"
        )


def render_home_dashboard() -> None:
    """Render the executive home dashboard for end users."""
    quality_path = PROCESSED_DATA_DIR / SETTINGS.quality_report_filename
    metrics_path = PROCESSED_DATA_DIR / SETTINGS.final_test_metrics_filename
    inference_metadata_path = MODEL_DIR / SETTINGS.inference_metadata_filename

    rows_used = 41757
    rows_original = 43824
    missing_target = 2067
    if quality_path.is_file():
        try:
            quality = load_json(str(quality_path))
            rows_original = int(quality["rows"])
            rows_used = int(quality["clean_rows"])
            missing_target = int(quality["missing_target"])
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            pass

    final_metrics: dict[str, object] | None = None
    if metrics_path.is_file():
        try:
            final_metrics = load_json(str(metrics_path))
        except (OSError, ValueError, json.JSONDecodeError):
            final_metrics = None

    st.markdown(
        """
        <div class="aq-hero">
          <div class="aq-eyebrow">AerQualitas · Deep Learning aplicado</div>
          <div class="aq-hero-title">Estimación interactiva de PM2.5 con una red neuronal feedforward</div>
          <p class="aq-hero-copy">
            AerQualitas integra datos meteorológicos históricos, un MLP validado y
            restricciones de entrada para producir estimaciones reproducibles de PM2.5
            que apoyan la evaluación de la calidad del aire.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_col, model_col, flow_col = st.columns([1, 1.35, 1])

    with dataset_col:
        st.markdown(
            f"""
            <div class="aq-card">
              <div class="aq-card-title">Dataset · Beijing PM2.5 (UCI)</div>
              <div class="aq-card-value">{rows_used:,} registros útiles</div>
              <div class="aq-card-copy">
                Fuente original: {rows_original:,} observaciones horarias (2010–2014).
                Se excluyen {missing_target:,} registros sin PM2.5 para el aprendizaje supervisado.
              </div>
              <div style="margin-top:.8rem">
                <span class="aq-pill">TEMP</span>
                <span class="aq-pill">PRES</span>
                <span class="aq-pill">DEWP</span>
                <span class="aq-pill">cbwd</span>
                <span class="aq-pill">Iws</span>
                <span class="aq-pill">Is</span>
                <span class="aq-pill">Ir</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with model_col:
        st.markdown(
            """
            <div class="aq-card">
              <div class="aq-card-title">Modelo DL utilizado · MLP feedforward</div>
              <div class="aq-card-copy">
                Perceptrón multicapa de regresión implementado en Python con TensorFlow/Keras.
                La dirección del viento se codifica antes de entrar a la red.
              </div>
              <div class="aq-architecture">
                <span class="aq-node">7 variables</span>
                <span class="aq-arrow">→</span>
                <span class="aq-node">10 características</span>
                <span class="aq-arrow">→</span>
                <span class="aq-node">Dense 64 · ReLU</span>
                <span class="aq-arrow">→</span>
                <span class="aq-node">Dense 32 · ReLU</span>
                <span class="aq-arrow">→</span>
                <span class="aq-node">Dense 1 · lineal</span>
              </div>
              <div class="aq-card-copy" style="margin-top:.9rem">
                Pérdida: MSE · Optimizador: Adam · Early stopping · Backpropagation
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with flow_col:
        st.markdown(
            """
            <div class="aq-card">
              <div class="aq-card-title">Cómo funciona</div>
              <div class="aq-flow-step"><span class="aq-step-number">1</span><span>Configure variables meteorológicas dentro de rangos permitidos.</span></div>
              <div class="aq-flow-step"><span class="aq-step-number">2</span><span>AerQualitas valida y preprocesa los datos.</span></div>
              <div class="aq-flow-step"><span class="aq-step-number">3</span><span>El MLP genera una estimación puntual de PM2.5.</span></div>
              <div class="aq-flow-step"><span class="aq-step-number">4</span><span>El resultado se presenta con trazabilidad del escenario evaluado.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="aq-section-kicker">Desempeño del modelo final</div>', unsafe_allow_html=True)
    if final_metrics is not None:
        mlp = final_metrics["mlp"]
        comparison = final_metrics["comparison"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE test", f"{float(mlp['mae']):.2f} µg/m³")
        m2.metric("RMSE test", f"{float(mlp['rmse']):.2f} µg/m³")
        m3.metric("R² test", f"{float(mlp['r2']):.3f}")
        m4.metric(
            "Frente al baseline",
            f"{int(comparison['metrics_improved'])}/3 métricas",
        )
        st.caption(
            "Métricas obtenidas sobre la partición temporal de prueba reservada. "
            "El test no se utiliza para reajustar el modelo."
        )
    else:
        st.info("Las métricas finales estarán disponibles después de ejecutar la evaluación del modelo.")

    inference_ready = inference_metadata_path.is_file()
    if inference_ready:
        st.success(
            "Motor de inferencia disponible. Puede estimar PM2.5 desde esta página "
            "o desde la sección Predicción."
        )
        render_prediction_section(compact=True)
    else:
        st.info(
            "El motor de inferencia no está preparado en este entorno. "
            "Ejecute `python prepare_inference.py --overwrite`."
        )

    with st.expander(
        "Fundamento del cálculo de PM2.5 en AerQualitas",
        expanded=False,
    ):
        st.markdown(
            """
            AerQualitas estima PM2.5 a partir de siete variables meteorológicas
            ingresadas por el usuario. Primero comprueba que los valores pertenezcan
            al dominio permitido y después aplica el mismo preprocesamiento utilizado
            durante el entrenamiento.
            """
        )
        render_function_refs(
            "validate_user_input()",
            "predict_pm25()",
        )

        st.markdown("#### Variables de entrada")
        st.latex(
            r"\mathbf{x}=[TEMP,\ PRES,\ DEWP,\ cbwd,\ Iws,\ Is,\ Ir]"
        )
        render_variable_refs(
            "TEMP",
            "PRES",
            "DEWP",
            "cbwd",
            "Iws",
            "Is",
            "Ir",
        )
        st.caption(
            "Estas siete variables representan el escenario meteorológico que "
            "AerQualitas enviará al pipeline de inferencia."
        )
        render_function_refs(
            "validate_user_input()",
        )

        st.markdown("#### Preprocesamiento")
        st.markdown(
            """
            Las seis variables numéricas se estandarizan mediante
            `StandardScaler()` y `cbwd` se transforma mediante **One-Hot Encoding**
            con `OneHotEncoder()`. Por ello, las **7 variables originales** se
            convierten en **10 características numéricas**.
            """
        )
        st.latex(
            r"\mathbf{x}'=\operatorname{Preprocesamiento}(\mathbf{x})"
        )
        render_variable_refs(
            "TEMP",
            "PRES",
            "DEWP",
            "Iws",
            "Is",
            "Ir",
            "cbwd",
            "x′",
        )
        render_function_refs(
            "build_preprocessor()",
            "StandardScaler()",
            "OneHotEncoder()",
            "preprocessor.transform()",
        )

        st.markdown("#### Función del MLP")
        st.latex(
            r"\widehat{PM2.5}=W_3\,"
            r"ReLU\!\left(W_2\,"
            r"ReLU\!\left(W_1\mathbf{x}'+b_1\right)+b_2\right)+b_3"
        )
        st.caption(
            "W representa los pesos y b los sesgos aprendidos durante el "
            "entrenamiento. ReLU introduce no linealidad en las capas ocultas."
        )
        render_variable_refs(
            "x′",
            "W₁",
            "W₂",
            "W₃",
            "b₁",
            "b₂",
            "b₃",
            "PM2.5 estimado",
        )
        render_function_refs(
            "build_mlp()",
            "fit_mlp()",
            "model.predict()",
        )

        st.markdown("#### Flujo del cálculo")
        st.code(
            "7 variables → validación → preprocesamiento → 10 características → "
            "Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, lineal) → "
            "PM2.5 estimado",
            language="text",
        )
        render_function_refs(
            "validate_user_input()",
            "preprocessor.transform()",
            "model.predict()",
            "predict_pm25()",
        )

        st.info(
            "AerQualitas no aplica una fórmula fija definida manualmente. "
            "La relación entre las variables meteorológicas y PM2.5 "
            "fue aprendida por la red neuronal durante el entrenamiento."
        )

        st.markdown("#### En síntesis")
        st.latex(
            r"\boxed{\text{Variables meteorológicas}"
            r"\;\rightarrow\;\text{Preprocesamiento}"
            r"\;\rightarrow\;\text{MLP entrenado}"
            r"\;\rightarrow\;\widehat{PM2.5}}"
        )
        st.caption(
            "Las variables ingresadas se validan y transforman antes de pasar "
            "al MLP entrenado, que produce la estimación final de PM2.5."
        )



def build_variable_dictionary() -> pd.DataFrame:
    """Return the user-facing dictionary for the variables used by AerQualitas."""
    return pd.DataFrame(
        [
            ["datetime", "Temporal", "Fecha y hora de la observación", "YYYY-MM-DD HH:00:00"],
            ["pm2.5", "Objetivo", "Concentración observada de material particulado fino PM2.5", "µg/m³"],
            ["DEWP", "Predictor", "Temperatura del punto de rocío", "°C"],
            ["TEMP", "Predictor", "Temperatura del aire", "°C"],
            ["PRES", "Predictor", "Presión atmosférica", "hPa"],
            ["cbwd", "Predictor", "Dirección combinada del viento", "NE, NW, SE, cv"],
            ["Iws", "Predictor", "Velocidad acumulada del viento registrada por el dataset", "m/s"],
            ["Is", "Predictor", "Horas acumuladas de nieve", "h"],
            ["Ir", "Predictor", "Horas acumuladas de lluvia", "h"],
        ],
        columns=["Variable", "Rol", "Descripción", "Unidad / valores"],
    )





def render_prediction_scatter(
    data: pd.DataFrame,
    *,
    actual_col: str,
    predicted_col: str,
    actual_label: str = "PM2.5 real",
    predicted_label: str = "PM2.5 estimado",
) -> None:
    """Render the native Streamlit scatter plot with unrestricted user zoom."""
    plot_data = data.loc[:, [actual_col, predicted_col]].dropna().copy()

    if plot_data.empty:
        st.info("No hay datos disponibles para construir esta visualización.")
        return

    st.scatter_chart(
        plot_data,
        x=actual_col,
        y=predicted_col,
        width="stretch",
        height=430,
    )
    st.caption(
        "Puede acercar, alejar y restablecer la vista libremente utilizando "
        "los controles interactivos del gráfico."
    )


def render_dataset_section() -> None:
    """Render a compact, technically grounded exploration of the Beijing PM2.5 dataset."""
    clean_path = PROCESSED_DATA_DIR / SETTINGS.processed_dataset_filename
    report_path = PROCESSED_DATA_DIR / SETTINGS.quality_report_filename

    st.subheader("Dataset Beijing PM2.5 — origen, estructura y exploración")

    if not clean_path.exists() or not report_path.exists():
        st.warning(
            "Los artefactos de datos procesados no están disponibles. Ejecute: "
            "`python prepare_data.py --overwrite`"
        )
        return

    try:
        data = load_csv(str(clean_path), parse_datetime=True)
        report = load_json(str(report_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"No fue posible cargar los artefactos procesados: {exc}")
        return

    original_rows = int(report["rows"])
    missing_target = int(report["missing_target"])
    clean_rows = int(report["clean_rows"])

    with st.container(border=True):
        st.markdown("### Ficha técnica y credenciales")
        st.markdown(
            """
            **Beijing PM2.5** es un conjunto de datos multivariado y de series de
            tiempo orientado a **regresión**. Contiene observaciones horarias entre
            **2010 y 2014**. La concentración de **PM2.5** procede de la
            **Embajada de Estados Unidos en Beijing**, mientras que las variables
            meteorológicas provienen del **Beijing Capital International Airport**.

            **Variable objetivo:** `pm2.5` (µg/m³).  
            **Predictores utilizados por AerQualitas:** `DEWP`, `TEMP`, `PRES`,
            `cbwd`, `Iws`, `Is`, `Ir`.  
            **Campos temporales:** `year`, `month`, `day`, `hour`; se conservan para
            orden cronológico y partición temporal, no como entradas directas al MLP.

            **Creador:** Song Chen · **Repositorio:** UCI Machine Learning Repository  
            **Referencia oficial:** Chen, S. (2015). *Beijing PM2.5* [Dataset].  
            **DOI:** [10.24432/C5JS49](https://doi.org/10.24432/C5JS49) ·
            **Licencia:** CC BY 4.0  
            **Fuente oficial:** [UCI — Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata)
            """
        )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Registros originales", f"{original_rows:,}")
    metric2.metric("PM2.5 faltantes", f"{missing_target:,}")
    metric3.metric("Registros utilizados", f"{clean_rows:,}")
    metric4.metric("Duplicados temporales", int(report["duplicate_timestamps"]))

    st.info(
        "**Aclaratoria:** AerQualitas utiliza únicamente los **41,757 registros "
        "con PM2.5 observado** para entrenamiento, validación y prueba. Las "
        f"**{missing_target:,} observaciones sin PM2.5** se excluyen porque no "
        "pueden actuar como respuesta supervisada. El archivo original se conserva "
        "sin modificaciones para trazabilidad y reproducibilidad."
    )

    tab_quality, tab_variables, tab_trend, tab_explorer = st.tabs(
        ["Calidad de datos", "Variables", "Tendencia PM2.5", "Explorador de datos"]
    )

    with tab_quality:
        st.markdown("#### Calidad estructural")
        quality_table = pd.DataFrame(
            {
                "Indicador": [
                    "Valores faltantes en predictores",
                    "Filas duplicadas",
                    "Timestamps duplicados",
                    "Saltos horarios en fuente original",
                ],
                "Valor": [
                    int(report["missing_features"]),
                    int(report["duplicate_rows"]),
                    int(report["duplicate_timestamps"]),
                    int(report["non_hourly_gaps"]),
                ],
            }
        )
        st.dataframe(quality_table, hide_index=True, width="stretch", height=180)
        st.caption(
            f"Periodo original: {str(report['datetime_min'])[:10]} a "
            f"{str(report['datetime_max'])[:10]}."
        )

    with tab_variables:
        st.markdown("#### Diccionario de variables utilizadas")
        st.dataframe(
            build_variable_dictionary(),
            hide_index=True,
            width="stretch",
            height=350,
        )

        categories = ", ".join(str(value) for value in report["wind_categories"])
        st.caption(f"Categorías observadas en `cbwd`: {categories}")

        st.markdown("#### Rangos observados en la fuente")
        ranges = report["numeric_ranges_raw"]
        range_rows = [
            {
                "Variable": variable,
                "Mínimo": values["min"],
                "Máximo": values["max"],
                "Media": round(values["mean"], 2),
                "Mediana": round(values["median"], 2),
            }
            for variable, values in ranges.items()
        ]
        st.dataframe(
            pd.DataFrame(range_rows),
            hide_index=True,
            width="stretch",
            height=300,
        )
        st.caption(
            "Estos rangos son descriptivos. Las restricciones interactivas de la "
            "app se derivan exclusivamente de la partición de entrenamiento."
        )

    with tab_trend:
        st.markdown("#### Evolución anual promedio de PM2.5")
        annual = (
            data.groupby("year", as_index=False)["pm2.5"]
            .mean()
            .rename(columns={"year": "Año", "pm2.5": "PM2.5 promedio"})
        )
        annual["Año"] = annual["Año"].astype(str)
        annual["PM2.5 promedio"] = annual["PM2.5 promedio"].round(2)

        chart = (
            alt.Chart(annual)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X(
                    "Año:N",
                    title="Año",
                    sort=["2010", "2011", "2012", "2013", "2014"],
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "PM2.5 promedio:Q",
                    title="PM2.5 promedio (µg/m³)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("Año:N", title="Año"),
                    alt.Tooltip("PM2.5 promedio:Q", title="PM2.5 promedio", format=".2f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")
        st.dataframe(annual, hide_index=True, width="stretch", height=215)
        st.caption(
            "Promedios anuales calculados sobre los registros limpios con PM2.5 "
            "observado. Esta visualización es descriptiva y no representa una "
            "predicción del MLP."
        )

    with tab_explorer:
        st.markdown("#### Explorador de datos")
        st.caption(
            "Los **41,757 registros limpios** pueden explorarse mediante filtros "
            "y paginación. La tabla renderiza solo la página actual para mantener "
            "la interfaz rápida y legible."
        )

        years = sorted(int(value) for value in data["year"].dropna().unique())
        months = sorted(int(value) for value in data["month"].dropna().unique())
        wind_categories = sorted(str(value) for value in data["cbwd"].dropna().unique())

        f1, f2, f3 = st.columns(3)
        with f1:
            selected_year = st.selectbox(
                "Año",
                ["Todos"] + [str(year) for year in years],
                key="explorer_year",
            )
        with f2:
            selected_month = st.selectbox(
                "Mes",
                ["Todos"] + [str(month) for month in months],
                key="explorer_month",
            )
        with f3:
            selected_wind = st.selectbox(
                "Dirección del viento",
                ["Todas"] + wind_categories,
                key="explorer_wind",
            )

        pm_min = float(data["pm2.5"].min())
        pm_max = float(data["pm2.5"].max())

        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            filter_pm_min = st.number_input(
                "PM2.5 mínimo (µg/m³)",
                min_value=pm_min,
                max_value=pm_max,
                value=pm_min,
                step=1.0,
                key="explorer_pm_min",
            )
        with r2:
            filter_pm_max = st.number_input(
                "PM2.5 máximo (µg/m³)",
                min_value=pm_min,
                max_value=pm_max,
                value=pm_max,
                step=1.0,
                key="explorer_pm_max",
            )
        with r3:
            rows_per_page = st.selectbox(
                "Filas por página",
                [25, 50, 100, 250],
                index=1,
                key="explorer_rows_per_page",
            )

        if filter_pm_min > filter_pm_max:
            st.error("El valor mínimo de PM2.5 no puede superar al máximo.")
            return

        filtered = data.copy()
        if selected_year != "Todos":
            filtered = filtered[filtered["year"] == int(selected_year)]
        if selected_month != "Todos":
            filtered = filtered[filtered["month"] == int(selected_month)]
        if selected_wind != "Todas":
            filtered = filtered[filtered["cbwd"] == selected_wind]

        filtered = filtered[
            filtered["pm2.5"].between(filter_pm_min, filter_pm_max, inclusive="both")
        ].reset_index(drop=True)

        total_records = len(filtered)
        total_pages = max(1, (total_records + rows_per_page - 1) // rows_per_page)

        if "dataset_explorer_page" not in st.session_state:
            st.session_state["dataset_explorer_page"] = 1
        st.session_state["dataset_explorer_page"] = min(
            max(1, int(st.session_state["dataset_explorer_page"])),
            total_pages,
        )

        nav1, nav2, nav3 = st.columns([1, 1, 2])
        with nav1:
            if st.button(
                "← Anterior",
                disabled=st.session_state["dataset_explorer_page"] <= 1,
                width="stretch",
            ):
                st.session_state["dataset_explorer_page"] -= 1
                st.rerun()
        with nav2:
            if st.button(
                "Siguiente →",
                disabled=st.session_state["dataset_explorer_page"] >= total_pages,
                width="stretch",
            ):
                st.session_state["dataset_explorer_page"] += 1
                st.rerun()
        with nav3:
            page = st.number_input(
                "Página",
                min_value=1,
                max_value=total_pages,
                step=1,
                key="dataset_explorer_page",
            )

        start = (int(page) - 1) * rows_per_page
        end = min(start + rows_per_page, total_records)

        st.markdown(
            f"**Registros {start + 1 if total_records else 0:,}–{end:,} "
            f"de {total_records:,}** · Página {int(page):,} de {total_pages:,}"
        )

        display_columns = [
            "datetime",
            "pm2.5",
            "DEWP",
            "TEMP",
            "PRES",
            "cbwd",
            "Iws",
            "Is",
            "Ir",
        ]
        available_columns = [
            column for column in display_columns if column in filtered.columns
        ]

        st.dataframe(
            filtered.loc[start:end - 1, available_columns] if total_records else filtered.loc[:, available_columns],
            hide_index=True,
            width="stretch",
            height=410,
        )

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Descargar vista filtrada CSV",
                data=filtered.loc[:, available_columns].to_csv(index=False).encode("utf-8"),
                file_name="aerqualitas_datos_filtrados.csv",
                mime="text/csv",
                width="stretch",
                disabled=total_records == 0,
            )
        with d2:
            st.download_button(
                "Descargar dataset limpio completo CSV",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="beijing_pm25_clean.csv",
                mime="text/csv",
                width="stretch",
            )

        st.markdown("---")
        st.markdown("#### Diccionario de variables")
        st.caption(
            "Referencia rápida para interpretar las columnas visibles en el "
            "explorador."
        )
        st.dataframe(
            build_variable_dictionary(),
            hide_index=True,
            width="stretch",
            height=350,
        )
        st.info(
            "`pm2.5` es la **variable objetivo**. `DEWP`, `TEMP`, `PRES`, "
            "`cbwd`, `Iws`, `Is` e `Ir` son las variables meteorológicas que "
            "alimentan el MLP después del preprocesamiento."
        )



def render_function_refs(*names: str) -> None:
    """Render implementation references with crimson function/class names."""
    crimson = "#DC143C"
    rendered = ", ".join(
        (
            f'<span style="color:{crimson};font-family:ui-monospace,'
            'SFMono-Regular,Menlo,Consolas,monospace;font-weight:650;">'
            f"{name}</span>"
        )
        for name in names
    )
    st.markdown(
        (
            '<div style="font-size:0.84rem;color:#617887;'
            'margin-top:-0.15rem;margin-bottom:0.75rem;">'
            '<strong>Funciones usadas:</strong> '
            f"{rendered}</div>"
        ),
        unsafe_allow_html=True,
    )



def render_variable_refs(*names: str) -> None:
    """Render variable and parameter references in crimson."""
    crimson = "#DC143C"
    rendered = ", ".join(
        (
            f'<span style="color:{crimson};font-family:ui-monospace,'
            'SFMono-Regular,Menlo,Consolas,monospace;font-weight:650;">'
            f"{name}</span>"
        )
        for name in names
    )
    st.markdown(
        (
            '<div style="font-size:0.84rem;color:#617887;'
            'margin-top:-0.15rem;margin-bottom:0.75rem;">'
            '<strong>Variables / parámetros:</strong> '
            f"{rendered}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_preprocessing_section() -> None:
    """Render chronological split, preprocessing and baseline results."""
    manifest_path = PROCESSED_DATA_DIR / SETTINGS.split_manifest_filename
    metrics_path = PROCESSED_DATA_DIR / SETTINGS.baseline_metrics_filename
    predictions_path = PROCESSED_DATA_DIR / SETTINGS.validation_predictions_filename

    if not manifest_path.exists() or not metrics_path.exists():
        st.warning(
            "Los artefactos de preprocesamiento aún no existen. Ejecute: "
            "`python train_baseline.py --overwrite`"
        )
        return

    try:
        manifest = load_json(str(manifest_path))
        metrics = load_json(str(metrics_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"No fue posible cargar los artefactos de preprocesamiento: {exc}")
        return

    st.subheader("Preprocesamiento y línea base")
    st.caption(
        "Prepara los datos, evita fugas de información y establece un modelo de "
        "referencia para comparar posteriormente el desempeño del MLP."
    )
    render_function_refs(
        "chronological_split()",
        "build_preprocessor()",
        "run_baseline_pipeline()",
    )
    st.markdown(
        "Los **41,757 registros limpios** se separan en orden temporal y sin "
        "barajar, evitando que observaciones futuras entrenen modelos destinados "
        "a periodos anteriores."
    )

    train = manifest["train"]
    validation = manifest["validation"]
    test = manifest["test"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Entrenamiento (70%)", f"{int(train['rows']):,}")
    c2.metric("Validación (15%)", f"{int(validation['rows']):,}")
    c3.metric("Prueba reservada (15%)", f"{int(test['rows']):,}")

    split_table = pd.DataFrame(
        [
            {
                "Partición": "Entrenamiento",
                "Registros": int(train["rows"]),
                "Inicio": str(train["datetime_start"])[:16],
                "Fin": str(train["datetime_end"])[:16],
                "Uso": "Ajustar preprocesamiento y modelos",
            },
            {
                "Partición": "Validación",
                "Registros": int(validation["rows"]),
                "Inicio": str(validation["datetime_start"])[:16],
                "Fin": str(validation["datetime_end"])[:16],
                "Uso": "Comparar y ajustar modelos",
            },
            {
                "Partición": "Prueba",
                "Registros": int(test["rows"]),
                "Inicio": str(test["datetime_start"])[:16],
                "Fin": str(test["datetime_end"])[:16],
                "Uso": "Reservada para evaluación final",
            },
        ]
    )
    st.dataframe(split_table, hide_index=True, width="stretch")

    st.warning(
        "El conjunto de **prueba permanece reservado** y no se usa "
        "para seleccionar arquitectura, hiperparámetros ni decisiones de modelado."
    )

    st.markdown("#### Transformaciones")
    st.caption(
        "Convierte las variables originales a una representación numérica adecuada "
        "y comparable para el entrenamiento de los modelos."
    )
    render_function_refs(
        "build_preprocessor()",
        "StandardScaler()",
        "OneHotEncoder()",
        "ColumnTransformer()",
    )
    transformation_table = pd.DataFrame(
        [
            {
                "Variables": "TEMP, PRES, DEWP, Iws, Is, Ir",
                "Tipo": "Numéricas",
                "Transformación": "StandardScaler",
            },
            {
                "Variables": "cbwd",
                "Tipo": "Categórica",
                "Transformación": "OneHotEncoder",
            },
        ]
    )
    st.dataframe(transformation_table, hide_index=True, width="stretch")
    st.caption(
        "El preprocesador se ajusta exclusivamente con entrenamiento y luego se "
        "reutiliza sin reajuste sobre validación, prueba e inferencia web."
    )

    st.markdown("#### Línea base: regresión lineal")
    st.caption(
        "Proporciona un modelo sencillo de referencia para determinar si la red "
        "neuronal aporta una mejora real."
    )
    render_function_refs(
        "build_linear_baseline()",
        "fit_linear_baseline()",
        "evaluate_pipeline()",
        "regression_metrics()",
    )

    train_metrics = metrics["train"]
    val_metrics = metrics["validation"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE validación", f"{float(val_metrics['mae']):.2f} µg/m³")
    m2.metric("RMSE validación", f"{float(val_metrics['rmse']):.2f} µg/m³")
    m3.metric("R² validación", f"{float(val_metrics['r2']):.3f}")

    comparison = pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "R²"],
            "Entrenamiento": [
                round(float(train_metrics["mae"]), 3),
                round(float(train_metrics["rmse"]), 3),
                round(float(train_metrics["r2"]), 3),
            ],
            "Validación": [
                round(float(val_metrics["mae"]), 3),
                round(float(val_metrics["rmse"]), 3),
                round(float(val_metrics["r2"]), 3),
            ],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    if predictions_path.exists():
        predictions = load_csv(str(predictions_path), parse_datetime=True)
        st.markdown("#### Valores reales frente a predichos — validación")
        st.caption(
            "Permite observar visualmente qué tan próximas se encuentran las "
            "estimaciones del modelo respecto a los valores reales de PM2.5."
        )
        render_function_refs(
            "validation_predictions()",
            "render_prediction_scatter()",
        )
        render_prediction_scatter(
            predictions,
            actual_col="pm25_real",
            predicted_col="pm25_predicho",
            actual_label="PM2.5 real",
            predicted_label="PM2.5 predicho",
        )

        st.markdown("#### Mayores errores absolutos de la línea base")
        st.caption(
            "Identifica los casos donde el modelo de referencia presenta las mayores "
            "diferencias entre el PM2.5 real y el estimado."
        )
        render_function_refs(
            "validation_predictions()",
            "DataFrame.nlargest()",
        )
        largest_errors = predictions.nlargest(10, "error_absoluto").copy()
        largest_errors["datetime"] = largest_errors["datetime"].astype(str)
        st.dataframe(largest_errors, hide_index=True, width="stretch")



def render_model_section() -> None:
    """Render model architecture, training history and validation comparison."""
    metrics_path = PROCESSED_DATA_DIR / SETTINGS.mlp_metrics_filename
    history_path = PROCESSED_DATA_DIR / SETTINGS.mlp_history_filename
    predictions_path = (
        PROCESSED_DATA_DIR / SETTINGS.mlp_validation_predictions_filename
    )
    metadata_path = MODEL_DIR / SETTINGS.mlp_metadata_filename

    st.subheader("Modelo Deep Learning")
    st.caption(
        "Presenta la red neuronal utilizada por AerQualitas y resume cómo fue "
        "configurada, entrenada y evaluada."
    )
    render_function_refs(
        "build_mlp()",
        "run_mlp_pipeline()",
    )
    st.markdown(
        "AerQualitas utiliza un **perceptrón multicapa (MLP) feedforward** "
        "para un problema de **regresión supervisada**. El modelo recibe las "
        "variables meteorológicas preprocesadas y estima una concentración continua "
        "de PM2.5."
    )

    st.code(
        "7 variables originales → preprocesamiento → entrada numérica → "
        "Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, lineal)",
        language="text",
    )

    st.caption(
        "La variable categórica cbwd se codifica con OneHotEncoder; por ello, "
        "la dimensión efectiva de entrada a la red puede ser mayor que siete."
    )

    architecture = pd.DataFrame(
        [
            {
                "Etapa": "Entrada",
                "Configuración": "Variables meteorológicas preprocesadas",
                "Propósito": "Representación numérica para la red",
            },
            {
                "Etapa": "Capa oculta 1",
                "Configuración": "Dense(64) + ReLU",
                "Propósito": "Aprender relaciones no lineales",
            },
            {
                "Etapa": "Capa oculta 2",
                "Configuración": "Dense(32) + ReLU",
                "Propósito": "Refinar representaciones",
            },
            {
                "Etapa": "Salida",
                "Configuración": "Dense(1) lineal",
                "Propósito": "Estimar PM2.5 continuo",
            },
        ]
    )
    st.dataframe(architecture, hide_index=True, width="stretch")

    training = pd.DataFrame(
        [
            {"Componente": "Función de pérdida", "Configuración": "MSE"},
            {"Componente": "Optimizador", "Configuración": "Adam"},
            {"Componente": "Mini-batch", "Configuración": str(SETTINGS.mlp_batch_size)},
            {
                "Componente": "Épocas máximas",
                "Configuración": str(SETTINGS.mlp_max_epochs),
            },
            {
                "Componente": "Early stopping",
                "Configuración": (
                    f"paciencia={SETTINGS.mlp_early_stopping_patience}"
                ),
            },
            {"Componente": "Semilla", "Configuración": str(SETTINGS.random_seed)},
        ]
    )
    st.markdown("#### Configuración de entrenamiento")
    st.caption(
        "Resume los principales parámetros utilizados para entrenar el MLP y "
        "controlar su aprendizaje."
    )
    render_function_refs(
        "MLPTrainingConfig",
        "configure_tensorflow()",
        "build_callbacks()",
        "fit_mlp()",
    )
    st.dataframe(training, hide_index=True, width="stretch")

    st.warning(
        "La selección del MLP se realizó exclusivamente con entrenamiento y "
        "validación. El conjunto de prueba solo puede emplearse en la evaluación "
        "final y no debe utilizarse para reajustar el modelo."
    )

    if not metrics_path.exists() or not metadata_path.exists():
        st.info(
            "El MLP aún no ha sido entrenado en este entorno. Ejecute en la terminal: "
            "`python train_mlp.py --overwrite`"
        )
        return

    try:
        metrics = load_json(str(metrics_path))
        metadata = load_json(str(metadata_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"No fue posible cargar los artefactos del MLP: {exc}")
        return

    val = metrics["validation"]
    baseline = metrics["baseline_validation"]
    comparison = metrics["comparison"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE MLP", f"{float(val['mae']):.2f} µg/m³")
    c2.metric("RMSE MLP", f"{float(val['rmse']):.2f} µg/m³")
    c3.metric("R² MLP", f"{float(val['r2']):.3f}")
    c4.metric(
        "Comparación baseline",
        "Mejora" if bool(comparison["beats_baseline"]) else "Sin mejora suficiente",
    )

    comparison_table = pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "R²"],
            "Baseline": [
                round(float(baseline["mae"]), 3),
                round(float(baseline["rmse"]), 3),
                round(float(baseline["r2"]), 3),
            ],
            "MLP": [
                round(float(val["mae"]), 3),
                round(float(val["rmse"]), 3),
                round(float(val["r2"]), 3),
            ],
        }
    )
    st.markdown("#### MLP frente a regresión lineal")
    st.caption(
        "Compara el desempeño de la red neuronal con el modelo de referencia "
        "para verificar si aporta una mejora real."
    )
    render_function_refs(
        "compare_with_baseline()",
        "evaluate_predictions()",
    )
    st.dataframe(comparison_table, hide_index=True, width="stretch")

    st.caption(
        f"Entrada efectiva a la red: {int(metadata['preprocessed_input_dim'])} "
        f"características. Épocas ejecutadas: {int(metadata['epochs_executed'])}."
    )

    if history_path.exists():
        history = load_csv(str(history_path))
        st.markdown("#### Curvas de entrenamiento")
        st.caption(
            "Muestran cómo evolucionan la pérdida y el error durante el "
            "entrenamiento y la validación."
        )
        render_function_refs(
            "fit_mlp()",
        )
        loss_columns = [column for column in ("loss", "val_loss") if column in history]
        if loss_columns:
            st.line_chart(history.set_index("epoch")[loss_columns])

    if predictions_path.exists():
        predictions = load_csv(str(predictions_path), parse_datetime=True)
        st.markdown("#### PM2.5 real frente a estimado — validación")
        st.caption(
            "Permite comparar visualmente los valores observados de PM2.5 con "
            "las estimaciones generadas por el MLP."
        )
        render_function_refs(
            "predict_array()",
            "prediction_frame()",
            "render_prediction_scatter()",
        )
        render_prediction_scatter(
            predictions,
            actual_col="pm25_real",
            predicted_col="pm25_predicho",
            actual_label="PM2.5 real",
            predicted_label="PM2.5 estimado",
        )

        st.markdown("#### Mayores errores absolutos del MLP")
        st.caption(
            "Identifica los casos de validación donde la red presenta las mayores "
            "diferencias entre el valor real y el estimado."
        )
        render_function_refs(
            "prediction_frame()",
            "DataFrame.nlargest()",
        )
        largest_errors = predictions.nlargest(10, "error_absoluto").copy()
        largest_errors["datetime"] = largest_errors["datetime"].astype(str)
        st.dataframe(largest_errors, hide_index=True, width="stretch")


def render_evaluation_section() -> None:
    """Render the final one-time evaluation on the reserved test set."""
    metrics_path = PROCESSED_DATA_DIR / SETTINGS.final_test_metrics_filename
    predictions_path = PROCESSED_DATA_DIR / SETTINGS.final_test_predictions_filename
    residual_path = PROCESSED_DATA_DIR / SETTINGS.final_error_analysis_filename
    quartile_path = PROCESSED_DATA_DIR / SETTINGS.final_error_by_quartile_filename
    manifest_path = MODEL_DIR / SETTINGS.final_model_manifest_filename

    st.subheader("Evaluación final")
    st.caption(
        "Resume el desempeño del MLP sobre el conjunto de prueba reservado y "
        "permite comprobar su capacidad de generalización."
    )
    render_function_refs(
        "run_final_evaluation()",
        "chronological_split()",
        "metrics_from_arrays()",
    )
    st.markdown(
        "El MLP fue seleccionado utilizando **solo validación**. "
        "Posteriormente se utiliza una sola vez la partición de **prueba reservada** "
        "para estimar el desempeño de generalización del modelo ya fijado."
    )
    st.warning(
        "Después de observar estas métricas, el conjunto de prueba no debe emplearse "
        "para ajustar capas, hiperparámetros o reglas del modelo. Hacerlo convertiría "
        "el test en otra partición de validación."
    )

    required = [metrics_path, predictions_path, residual_path, quartile_path, manifest_path]
    if not all(path.exists() for path in required):
        st.info(
            "La evaluación final aún no se ha ejecutado en este entorno. Ejecute "
            "una sola vez: `python evaluate_final.py`"
        )
        return

    try:
        metrics = load_json(str(metrics_path))
        residuals = load_json(str(residual_path))
        manifest = load_json(str(manifest_path))
        predictions = load_csv(str(predictions_path), parse_datetime=True)
        quartiles = load_csv(str(quartile_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"No fue posible cargar los artefactos de evaluación final: {exc}")
        return

    mlp = metrics["mlp"]
    baseline = metrics["baseline"]
    comparison = metrics["comparison"]
    validation = metrics["validation_reference_mlp"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE test MLP", f"{float(mlp['mae']):.2f} µg/m³")
    c2.metric("RMSE test MLP", f"{float(mlp['rmse']):.2f} µg/m³")
    c3.metric("R² test MLP", f"{float(mlp['r2']):.3f}")
    c4.metric(
        "Frente a baseline",
        f"{int(comparison['metrics_improved'])}/3 métricas",
    )

    st.markdown("#### Comparación final: MLP vs. regresión lineal")
    st.caption(
        "Contrasta ambos modelos sobre los mismos datos de prueba para verificar "
        "cuál ofrece mejor desempeño."
    )
    render_function_refs(
        "compare_test_metrics()",
        "metrics_from_arrays()",
    )
    comparison_table = pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "R²"],
            "Baseline test": [
                round(float(baseline["mae"]), 3),
                round(float(baseline["rmse"]), 3),
                round(float(baseline["r2"]), 3),
            ],
            "MLP test": [
                round(float(mlp["mae"]), 3),
                round(float(mlp["rmse"]), 3),
                round(float(mlp["r2"]), 3),
            ],
        }
    )
    st.dataframe(comparison_table, hide_index=True, width="stretch")

    st.markdown("#### Generalización: validación frente a prueba")
    st.caption(
        "Permite comprobar si el rendimiento observado durante validación se "
        "mantiene sobre datos no utilizados previamente."
    )
    render_function_refs(
        "run_final_evaluation()",
        "load_json()",
    )
    generalization = pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "R²"],
            "Validación MLP": [
                round(float(validation["mae"]), 3),
                round(float(validation["rmse"]), 3),
                round(float(validation["r2"]), 3),
            ],
            "Test MLP": [
                round(float(mlp["mae"]), 3),
                round(float(mlp["rmse"]), 3),
                round(float(mlp["r2"]), 3),
            ],
        }
    )
    st.dataframe(generalization, hide_index=True, width="stretch")
    st.caption(
        f"Periodo de prueba: {str(metrics['test_period']['start'])[:16]} a "
        f"{str(metrics['test_period']['end'])[:16]} — "
        f"{int(metrics['test_rows']):,} registros."
    )

    st.markdown("#### Análisis de errores del MLP")
    st.caption(
        "Resume la magnitud y dirección de los errores para detectar tendencias "
        "de sobreestimación o subestimación."
    )
    render_function_refs(
        "residual_analysis()",
    )
    mlp_error = residuals["mlp"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Residuo medio", f"{float(mlp_error['mean_residual']):.2f}")
    e2.metric(
        "Error absoluto mediano",
        f"{float(mlp_error['median_absolute_error']):.2f}",
    )
    e3.metric(
        "P90 error absoluto",
        f"{float(mlp_error['p90_absolute_error']):.2f}",
    )
    e4.metric(
        "Error absoluto máximo",
        f"{float(mlp_error['max_absolute_error']):.2f}",
    )
    st.caption(
        "Residuo = PM2.5 observado − PM2.5 estimado. Un residuo medio positivo "
        "indica subestimación promedio; uno negativo, sobreestimación promedio."
    )

    st.markdown("#### Error por cuartiles del PM2.5 observado")
    st.caption(
        "Muestra cómo varía el error del modelo según diferentes niveles de "
        "concentración real de PM2.5."
    )
    render_function_refs(
        "error_by_target_quartile()",
        "pandas.qcut()",
    )
    st.dataframe(quartiles, hide_index=True, width="stretch")
    st.caption(
        "Q1–Q4 son cuartiles descriptivos del test y no representan categorías "
        "oficiales de calidad del aire."
    )

    st.markdown("#### PM2.5 real frente a estimado — test")
    st.caption(
        "Permite comparar visualmente los valores observados de PM2.5 con las "
        "estimaciones finales generadas por el MLP."
    )
    render_function_refs(
        "final_prediction_frame()",
        "render_prediction_scatter()",
    )
    render_prediction_scatter(
        predictions,
        actual_col="pm25_real",
        predicted_col="pm25_mlp",
        actual_label="PM2.5 real",
        predicted_label="PM2.5 estimado",
    )

    st.markdown("#### Casos con mayor error absoluto del MLP")
    st.caption(
        "Identifica los casos donde el MLP presenta los errores de predicción más "
        "altos, ya sea por sobreestimar o subestimar el PM2.5 real."
    )
    render_function_refs(
        "final_prediction_frame()",
        "DataFrame.nlargest()",
    )
    largest = predictions.nlargest(12, "error_absoluto_mlp").copy()
    largest["datetime"] = largest["datetime"].astype(str)
    st.dataframe(largest, hide_index=True, width="stretch")

    st.success(
        "Modelo consolidado: **MLP feedforward**. La selección se decidió antes "
        "de abrir el test; esta sección documenta únicamente su desempeño final."
    )

    with st.expander("Trazabilidad del modelo final"):
        st.json(
            {
                "estado": manifest["status"],
                "modelo": manifest["selected_model"],
                "selección_previa_al_test": manifest[
                    "selection_decided_before_test"
                ],
                "ajuste_posterior_con_test": manifest["post_test_tuning_allowed"],
                "sha256": manifest["artifact_sha256"],
            }
        )




@st.cache_data(show_spinner=False)
def build_training_presets(
    clean_dataset_path: str,
    train_fraction: float,
) -> dict[str, dict[str, object]]:
    """Build four representative scenarios from the training partition only."""
    data = load_csv(clean_dataset_path, parse_datetime=True)
    train_end = int(len(data) * train_fraction)
    train = data.iloc[:train_end].copy()

    if train.empty or "pm2.5" not in train.columns:
        raise ValueError("No hay datos de entrenamiento suficientes para generar ejemplos.")

    definitions = [
        ("Muy desfavorable", 0.95, "🔴"),
        ("Desfavorable", 0.75, "🟠"),
        ("Intermedio", 0.50, "🟡"),
        ("Favorable", 0.10, "🟢"),
    ]

    presets: dict[str, dict[str, object]] = {}
    for label, quantile, icon in definitions:
        target = float(train["pm2.5"].quantile(quantile))
        nearest_index = (train["pm2.5"] - target).abs().idxmin()
        row = train.loc[nearest_index]

        presets[label] = {
            "label": label,
            "icon": icon,
            "reference_pm25": float(row["pm2.5"]),
            "datetime": str(row["datetime"]),
            "payload": {
                "TEMP": float(row["TEMP"]),
                "PRES": float(row["PRES"]),
                "DEWP": float(row["DEWP"]),
                "cbwd": str(row["cbwd"]),
                "Iws": float(row["Iws"]),
                "Is": float(row["Is"]),
                "Ir": float(row["Ir"]),
            },
        }

    return presets


def initialize_prediction_state(
    numeric: dict[str, object],
    categorical: dict[str, object],
) -> None:
    """Initialize and clamp prediction widgets to the allowed inference domain."""
    for key, spec in numeric.items():
        state_key = f"aq_input_{key}"
        lower = float(spec["min"])
        upper = float(spec["max"])
        default = float(spec["default"])

        current = float(st.session_state.get(state_key, default))
        st.session_state[state_key] = min(max(current, lower), upper)

    allowed = [str(value) for value in categorical["allowed"]]
    default_category = str(categorical["default"])
    category_key = "aq_input_cbwd"
    current_category = str(
        st.session_state.get(category_key, default_category)
    )
    st.session_state[category_key] = (
        current_category if current_category in allowed else default_category
    )


def prediction_payload_matches_preset(
    payload: dict[str, object],
    preset_payload: dict[str, object] | None,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return True only when the current inputs still match the loaded preset."""
    if not isinstance(preset_payload, dict):
        return False

    numeric_keys = ("TEMP", "PRES", "DEWP", "Iws", "Is", "Ir")
    try:
        for key in numeric_keys:
            if abs(float(payload[key]) - float(preset_payload[key])) > tolerance:
                return False
        return str(payload["cbwd"]) == str(preset_payload["cbwd"])
    except (KeyError, TypeError, ValueError):
        return False


def load_preset_into_prediction_state(
    preset: dict[str, object],
    *,
    numeric: dict[str, object],
    categorical: dict[str, object],
) -> None:
    """Load a training-derived preset into prediction widget state."""
    payload = dict(preset["payload"])

    for key, spec in numeric.items():
        lower = float(spec["min"])
        upper = float(spec["max"])
        value = float(payload[key])
        st.session_state[f"aq_input_{key}"] = min(max(value, lower), upper)

    allowed = [str(value) for value in categorical["allowed"]]
    category = str(payload["cbwd"])
    if category not in allowed:
        category = str(categorical["default"])
    st.session_state["aq_input_cbwd"] = category

    st.session_state["aq_selected_preset_label"] = str(preset["label"])
    st.session_state["aq_selected_preset_reference_pm25"] = float(
        preset["reference_pm25"]
    )
    st.session_state["aq_selected_preset_datetime"] = str(preset["datetime"])
    st.session_state["aq_selected_preset_payload"] = {
        key: float(payload[key]) for key in numeric
    }
    st.session_state["aq_selected_preset_payload"]["cbwd"] = category


def render_prediction_section(*, compact: bool = False) -> None:
    """Render interactive inference with manual and preset scenarios."""
    metadata_path = MODEL_DIR / SETTINGS.inference_metadata_filename
    model_path = MODEL_DIR / SETTINGS.mlp_model_filename
    preprocessor_path = MODEL_DIR / SETTINGS.preprocessor_filename
    manifest_path = MODEL_DIR / SETTINGS.final_model_manifest_filename
    clean_path = PROCESSED_DATA_DIR / SETTINGS.processed_dataset_filename

    if compact:
        st.markdown("### Estimación rápida de PM2.5")
        st.caption(
            "Configure manualmente un escenario o cargue uno de los ejemplos "
            "históricos preestablecidos y luego ejecute la estimación."
        )
    else:
        st.subheader("Estimación interactiva de PM2.5")
        st.caption(
            "Puede introducir condiciones meteorológicas manualmente o utilizar "
            "un ejemplo histórico preestablecido. En ambos casos, la predicción "
            "se realiza con el mismo MLP validado."
        )

    required = [
        metadata_path,
        model_path,
        preprocessor_path,
        manifest_path,
        clean_path,
    ]
    if not all(path.is_file() for path in required):
        st.info(
            "El motor de inferencia o los datos preparados aún no están disponibles "
            "en este entorno. Ejecute `python prepare_inference.py --overwrite`."
        )
        return

    try:
        metadata = load_inference_metadata(metadata_path)
        presets = build_training_presets(
            str(clean_path),
            SETTINGS.train_fraction,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"No fue posible preparar la inferencia interactiva: {exc}")
        return

    st.info(
        "Restricción intrínseca activa: AerQualitas **no permite extrapolar** "
        "fuera de los rangos derivados exclusivamente de la partición de entrenamiento."
    )

    numeric = metadata["numeric"]
    categorical = metadata["categorical"]["cbwd"]
    initialize_prediction_state(numeric, categorical)

    manual_tab, presets_tab = st.tabs(
        ["Entrada manual", "Ejemplos preestablecidos"]
    )

    # The preset tab is rendered first in code so a button can update state
    # before the widgets are instantiated in this run.
    with presets_tab:
        st.markdown("#### Escenarios históricos orientativos")
        st.caption(
            "Estos ejemplos se seleccionan automáticamente de la **partición de "
            "entrenamiento** según diferentes niveles históricos de PM2.5. "
            "Las etiquetas son didácticas y **no representan categorías AQI oficiales**."
        )

        preset_items = list(presets.items())
        rows = [preset_items[:2], preset_items[2:]]

        for preset_row in rows:
            columns = st.columns(2)
            for column, (name, preset) in zip(columns, preset_row):
                with column:
                    with st.container(border=True):
                        st.markdown(
                            f"### {preset['icon']} {name}"
                        )
                        st.metric(
                            "PM2.5 histórico de referencia",
                            f"{float(preset['reference_pm25']):.1f} µg/m³",
                        )
                        payload = preset["payload"]
                        st.caption(
                            f"TEMP {float(payload['TEMP']):.0f} °C · "
                            f"PRES {float(payload['PRES']):.0f} hPa · "
                            f"DEWP {float(payload['DEWP']):.0f} °C · "
                            f"Viento {str(payload['cbwd'])}"
                        )
                        st.caption(
                            f"Iws {float(payload['Iws']):.2f} · "
                            f"Is {float(payload['Is']):.0f} h · "
                            f"Ir {float(payload['Ir']):.0f} h"
                        )
                        if st.button(
                            "Cargar ejemplo",
                            key=f"load_preset_{name}",
                            width="stretch",
                        ):
                            load_preset_into_prediction_state(
                                preset,
                                numeric=numeric,
                                categorical=categorical,
                            )
                            st.rerun()

        st.info(
            "Después de cargar un ejemplo puede cambiar cualquier valor dentro de "
            "los rangos permitidos antes de pulsar **Estimar PM2.5**."
        )

    with manual_tab:
        selected_label = st.session_state.get("aq_selected_preset_label")
        preset_status = st.empty()

        col1, col2 = st.columns(2)

        with col1:
            temp_spec = numeric["TEMP"]
            temp = st.slider(
                f"{temp_spec['label']} ({temp_spec['unit']})",
                min_value=float(temp_spec["min"]),
                max_value=float(temp_spec["max"]),
                step=float(temp_spec["step"]),
                help=temp_spec["description"],
                key="aq_input_TEMP",
            )

            dewp_spec = numeric["DEWP"]
            dewp = st.slider(
                f"{dewp_spec['label']} ({dewp_spec['unit']})",
                min_value=float(dewp_spec["min"]),
                max_value=float(dewp_spec["max"]),
                step=float(dewp_spec["step"]),
                help=dewp_spec["description"],
                key="aq_input_DEWP",
            )

            iws_spec = numeric["Iws"]
            iws = st.number_input(
                f"{iws_spec['label']} ({iws_spec['unit']})",
                min_value=float(iws_spec["min"]),
                max_value=float(iws_spec["max"]),
                step=float(iws_spec["step"]),
                help=iws_spec["description"],
                key="aq_input_Iws",
            )

        with col2:
            pres_spec = numeric["PRES"]
            pres = st.slider(
                f"{pres_spec['label']} ({pres_spec['unit']})",
                min_value=float(pres_spec["min"]),
                max_value=float(pres_spec["max"]),
                step=float(pres_spec["step"]),
                help=pres_spec["description"],
                key="aq_input_PRES",
            )

            is_spec = numeric["Is"]
            snow = st.number_input(
                f"{is_spec['label']} ({is_spec['unit']})",
                min_value=float(is_spec["min"]),
                max_value=float(is_spec["max"]),
                step=float(is_spec["step"]),
                help=is_spec["description"],
                key="aq_input_Is",
            )

            ir_spec = numeric["Ir"]
            rain = st.number_input(
                f"{ir_spec['label']} ({ir_spec['unit']})",
                min_value=float(ir_spec["min"]),
                max_value=float(ir_spec["max"]),
                step=float(ir_spec["step"]),
                help=ir_spec["description"],
                key="aq_input_Ir",
            )

        allowed = [str(value) for value in categorical["allowed"]]
        cbwd = st.selectbox(
            categorical["label"],
            options=allowed,
            help=categorical["description"],
            key="aq_input_cbwd",
        )

        payload = {
            "TEMP": temp,
            "PRES": pres,
            "DEWP": dewp,
            "cbwd": cbwd,
            "Iws": iws,
            "Is": snow,
            "Ir": rain,
        }

        preset_reference_active = bool(selected_label) and prediction_payload_matches_preset(
            payload,
            st.session_state.get("aq_selected_preset_payload"),
        )

        if selected_label and preset_reference_active:
            reference = float(
                st.session_state["aq_selected_preset_reference_pm25"]
            )
            preset_status.success(
                f"Ejemplo cargado: **{selected_label}** · PM2.5 histórico de "
                f"referencia: **{reference:.1f} µg/m³**."
            )
        elif selected_label:
            preset_status.info(
                "Los valores del ejemplo cargado fueron modificados. "
                "La referencia histórica del preset ya no se utilizará para "
                "comparar esta estimación."
            )

        if st.button(
            "Estimar PM2.5",
            type="primary",
            width="stretch",
            key="estimate_pm25_button",
        ):
            try:
                assets = load_final_inference_assets(
                    str(model_path),
                    str(preprocessor_path),
                    str(metadata_path),
                    str(manifest_path),
                )
                result = predict_pm25(
                    payload,
                    model=assets.model,
                    preprocessor=assets.preprocessor,
                    metadata=assets.metadata,
                )
            except InputValidationError as exc:
                st.error(f"Entrada rechazada: {exc}")
                return
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"No fue posible realizar la predicción: {exc}")
                return

            if result.physical_warning:
                st.warning(result.physical_warning)

            st.metric(
                "PM2.5 estimado",
                f"{result.pm25:.2f} µg/m³",
            )
            st.caption(
                f"La inferencia utilizó {result.transformed_features} "
                "características después del preprocesamiento."
            )

            if preset_reference_active:
                reference = float(
                    st.session_state["aq_selected_preset_reference_pm25"]
                )
                difference = result.pm25 - reference

                if difference > 0:
                    interpretation = "sobreestimación"
                elif difference < 0:
                    interpretation = "subestimación"
                else:
                    interpretation = "estimación exacta"

                st.markdown(
                    (
                        f"**Referencia histórica:** {reference:.2f} µg/m³ · "
                        f"**Estimación MLP:** {result.pm25:.2f} µg/m³ · "
                        f"**Diferencia:** {difference:+.2f} µg/m³ "
                        f"(**{interpretation}**)"
                    )
                )
                st.caption(
                    "La referencia histórica corresponde al registro real usado "
                    "como ejemplo preestablecido. La diferencia muestra cuánto se "
                    "aleja la estimación del MLP de ese caso concreto."
                )

            st.markdown("#### Escenario evaluado")
            scenario = pd.DataFrame(
                [
                    {
                        "Variable": (
                            numeric[key]["label"]
                            if key in numeric
                            else categorical["label"]
                        ),
                        "Valor": (
                            f"{float(value):g}"
                            if key in numeric
                            else str(value)
                        ),
                        "Unidad": (
                            str(numeric[key]["unit"])
                            if key in numeric
                            else "código"
                        ),
                    }
                    for key, value in payload.items()
                ]
            )
            # Arrow requiere un tipo homogéneo por columna. "Valor" contiene
            # tanto magnitudes numéricas como la categoría del viento, por lo que
            # se presenta explícitamente como texto para evitar conversiones ambiguas.
            scenario = scenario.astype(
                {
                    "Variable": "string",
                    "Valor": "string",
                    "Unidad": "string",
                }
            )
            st.dataframe(
                scenario,
                hide_index=True,
                width="stretch",
            )

            st.info(
                "La estimación de PM2.5 se utiliza como **indicador para apoyar "
                "la evaluación de la calidad del aire**. No constituye una "
                "medición física ni un AQI oficial. El AQI de PM2.5 requiere "
                "una concentración promedio de 24 horas."
            )

        if not compact:
            with st.expander("Rangos intrínsecos permitidos"):
                rows = [
                    {
                        "Variable": spec["label"],
                        "Mínimo": spec["min"],
                        "Máximo": spec["max"],
                        "Unidad": spec["unit"],
                    }
                    for spec in numeric.values()
                ]
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    width="stretch",
                )
                st.write(
                    "Direcciones permitidas:",
                    ", ".join(allowed),
                )




def render_help_section() -> None:
    """Render concise help, provenance, model explanation and glossary."""
    st.subheader("Ayuda y glosario")
    st.markdown(
        "Esta sección resume qué hace AerQualitas, de dónde provienen sus datos, "
        "cómo participa el modelo Deep Learning y qué significan los términos "
        "principales utilizados en la aplicación."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Cómo funciona", "Dataset", "Modelo DL", "Glosario"]
    )

    with tab1:
        st.markdown("#### Flujo de uso")
        st.code(
            "Escenario meteorológico → Validación de rangos → Preprocesamiento → "
            "MLP feedforward → PM2.5 estimado",
            language="text",
        )
        st.markdown(
            """
            1. El usuario configura siete variables meteorológicas dentro de los
               rangos permitidos.
            2. AerQualitas vuelve a validar las entradas en el backend.
            3. El preprocesador aplica exactamente las transformaciones utilizadas
               durante el entrenamiento.
            4. El MLP consolidado genera una estimación puntual de PM2.5.
            5. El resultado se presenta junto con el escenario que produjo la
               inferencia.
            """
        )
        st.info(
            "AerQualitas **estima PM2.5** para apoyar la evaluación de la calidad "
            "del aire. No funciona como sensor físico y no presenta la salida puntual "
            "como un AQI oficial."
        )

    with tab2:
        st.markdown("#### Origen y aporte del dataset")
        st.markdown(
            """
            Se utiliza **Beijing PM2.5**, publicado en el **UCI Machine Learning
            Repository**. La fuente contiene **43,824 registros horarios de
            2010–2014**. Para aprendizaje supervisado se excluyen **2,067 registros
            sin PM2.5**, quedando **41,757 observaciones** para entrenamiento,
            validación y prueba.

            Las variables meteorológicas aportan las entradas del modelo y el
            PM2.5 observado aporta la **variable objetivo** con la que la red aprende
            a reducir el error de sus estimaciones.
            """
        )
        st.caption(
            "Referencia: Chen, S. (2015). Beijing PM2.5 [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49"
        )

    with tab3:
        st.markdown("#### Modelo Deep Learning utilizado")
        st.code(
            "7 variables → preprocesamiento → 10 características → "
            "Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, lineal)",
            language="text",
        )
        st.markdown(
            """
            **Tipo:** perceptrón multicapa (**MLP**) con arquitectura
            **feedforward** para regresión supervisada.

            **Entrenamiento:** pérdida MSE, optimizador Adam, mini-batches,
            backpropagation, reducción adaptativa de la tasa de aprendizaje y
            early stopping.

            **Evaluación final sobre test reservado:** MAE 47.782 µg/m³,
            RMSE 64.584 µg/m³ y R² 0.333. El MLP superó a la regresión lineal en
            las tres métricas evaluadas.
            """
        )

    with tab4:
        glossary = pd.DataFrame(
            [
                ("PM2.5", "Material particulado fino con diámetro aerodinámico ≤ 2.5 µm."),
                ("DL", "Deep Learning o aprendizaje profundo."),
                ("MLP", "Multilayer Perceptron; perceptrón multicapa feedforward."),
                ("ReLU", "Rectified Linear Unit; función de activación usada en capas ocultas."),
                ("MSE", "Mean Squared Error; error cuadrático medio usado como pérdida."),
                ("MAE", "Mean Absolute Error; error absoluto medio."),
                ("RMSE", "Root Mean Squared Error; raíz del error cuadrático medio."),
                ("R²", "Coeficiente de determinación."),
                ("Adam", "Optimizador adaptativo que actualiza los parámetros del MLP."),
                ("AQI", "Air Quality Index; índice de calidad del aire."),
                ("Baseline", "Modelo de referencia sencillo usado para comparar desempeño."),
                ("One-Hot", "Codificación numérica aplicada a la dirección del viento."),
            ],
            columns=["Término", "Definición breve"],
        )
        st.dataframe(glossary, hide_index=True, width="stretch")


def render_footer() -> None:
    """Render a compact project footer."""
    st.markdown(
        """
        <div class="aq-footer">
        <strong>AerQualitas</strong> · MLP feedforward en Python/TensorFlow-Keras ·
        Interfaz Streamlit · Inferencia restringida al dominio de entrenamiento.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scope() -> None:
    """Render the planned project pipeline."""
    st.subheader("Pipeline del proyecto")
    st.code(
        "Dataset → Validación → Preprocesamiento → Baseline → MLP feedforward → "
        "Entrenamiento → Evaluación → Inferencia interactiva → Despliegue web",
        language="text",
    )
    st.markdown(
        """
        **Objetivo:** estimar la concentración de PM2.5 a partir de variables
        meteorológicas históricas mediante un MLP feedforward y presentar la
        inferencia de forma interactiva, documentada y reproducible.
        """
    )



def render_institutional_section() -> None:
    """Render the academic and institutional identification of AerQualitas."""
    st.subheader("Institucional")
    st.caption(
        "Identificación académica del proyecto AerQualitas y del contexto "
        "formativo en el que fue desarrollado."
    )

    st.markdown(
        """
        <div class="aq-hero">
          <div class="aq-eyebrow">Proyecto académico</div>
          <div class="aq-hero-title">
            <a href="https://www.iinternacional.edu.mx/"
               target="_blank"
               rel="noopener noreferrer"
               style="color:inherit;text-decoration:none;">
              INSTITUTO INTERNACIONAL DE AGUASCALIENTES
            </a>
          </div>
          <p class="aq-hero-copy">
            Maestría en Inteligencia Artificial para la Transformación Digital
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.link_button(
        "Sitio oficial del Instituto Internacional de Aguascalientes",
        "https://www.iinternacional.edu.mx/",
        width="content",
    )

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### Información académica")
            st.markdown("**Asignatura:** Aprendizaje Profundo")
            st.markdown("**Aplicación:** AerQualitas")
            st.markdown(
                "**Proyecto:** Diseño, implementación y despliegue web de una "
                "red neuronal feedforward en Python para la estimación de PM2.5 "
                "a partir de variables meteorológicas históricas."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### Autoría académica")
            st.markdown("**Alumno:** Antonio Nicolás Toro González")
            st.markdown("**Tutora:** Dra. Claudia Andrea Vidales Basurto")

    st.markdown("### Descripción del proyecto")
    st.markdown(
        """
        AerQualitas es una aplicación web interactiva desarrollada en Python que
        utiliza una red neuronal **MLP feedforward** para estimar concentraciones
        de **PM2.5** a partir de variables meteorológicas históricas. El proyecto
        integra preparación de datos, preprocesamiento, entrenamiento, evaluación,
        análisis de errores, inferencia y despliegue web.
        """
    )

    st.markdown("### Identificación técnica")
    technical = pd.DataFrame(
        [
            ["Lenguaje", "Python"],
            ["Modelo", "MLP feedforward para regresión"],
            ["Deep Learning", "TensorFlow / Keras"],
            ["Preprocesamiento", "scikit-learn"],
            ["Datos", "pandas"],
            ["Interfaz web", "Streamlit"],
            ["Dataset", "Beijing PM2.5 — UCI Machine Learning Repository"],
        ],
        columns=["Componente", "Tecnología / referencia"],
    )
    st.dataframe(
        technical,
        hide_index=True,
        width="stretch",
        height=285,
    )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title=SETTINGS.name,
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_visual_style()

    logger = configure_logging()
    logger.info("Inicio de AerQualitas %s", SETTINGS.version)

    render_header()
    st.divider()

    st.sidebar.markdown("### AerQualitas")
    st.sidebar.caption("Evaluación de calidad del aire mediante estimación de PM2.5")
    page = st.sidebar.radio(
        "Navegación",
        (
            "Inicio",
            "Dataset",
            "Preprocesamiento",
            "Modelo DL",
            "Evaluación",
            "Predicción",
            "Ayuda",
            "Institucional",
        ),
    )

    if page == "Inicio":
        render_home_dashboard()
    elif page == "Dataset":
        render_dataset_section()
    elif page == "Preprocesamiento":
        render_preprocessing_section()
    elif page == "Modelo DL":
        render_model_section()
    elif page == "Evaluación":
        render_evaluation_section()
    elif page == "Predicción":
        render_prediction_section()
    elif page == "Ayuda":
        render_help_section()
    else:
        render_institutional_section()

    render_footer()


if __name__ == "__main__":
    main()
