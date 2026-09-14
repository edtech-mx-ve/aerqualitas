<div align="center">

<img src="assets/logo.png" alt="AerQualitas" width="120"/>

# AerQualitas

### Estimación de PM2.5 con Deep Learning a partir de variables meteorológicas

Aplicación web interactiva desarrollada en **Python** con un **MLP feedforward** para estimar concentraciones de PM2.5 y apoyar la evaluación de la calidad del aire.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI-0A66C2)](https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata)
[![App](https://img.shields.io/badge/Streamlit-AerQualitas%20Online-FF4B4B?logo=streamlit&logoColor=white)](https://aerqualitas.streamlit.app/)

**Instituto Internacional de Aguascalientes**  
Maestría en Inteligencia Artificial para la Transformación Digital · Aprendizaje Profundo

[🚀 Abrir AerQualitas](https://aerqualitas.streamlit.app/) ·
[Repositorio](https://github.com/edtech-mx-ve/aerqualitas) ·
[Institución](https://www.iinternacional.edu.mx/) ·
[Dataset UCI](https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata)

</div>

---

## Aplicación en línea

AerQualitas está desplegada públicamente en **Streamlit Community Cloud** y puede utilizarse directamente desde el navegador:

### 👉 [https://aerqualitas.streamlit.app/](https://aerqualitas.streamlit.app/)

No requiere instalación local para realizar estimaciones. La aplicación utiliza el **modelo MLP ya entrenado**, junto con su preprocesador y metadatos de inferencia versionados en el repositorio.

---

## ¿Qué hace AerQualitas?

AerQualitas estima la concentración de **PM2.5** a partir de siete variables meteorológicas históricas:

`TEMP` · `PRES` · `DEWP` · `cbwd` · `Iws` · `Is` · `Ir`

La aplicación:

- valida que los valores de entrada estén dentro del dominio conocido por el modelo;
- aplica el mismo preprocesamiento usado durante el entrenamiento;
- transforma 7 variables originales en 10 características numéricas;
- ejecuta una red neuronal **MLP feedforward**;
- devuelve una estimación puntual de PM2.5 en `µg/m³`;
- permite explorar dataset, preprocesamiento, modelo, evaluación, predicción y ayuda técnica.

> **Importante:** AerQualitas no mide físicamente la calidad del aire y no calcula un AQI oficial. Produce una estimación de PM2.5 basada en un modelo entrenado con datos históricos.

---

## Flujo general

```text
Variables meteorológicas
        ↓
Validación
        ↓
Preprocesamiento
        ↓
10 características numéricas
        ↓
Dense(64, ReLU)
        ↓
Dense(32, ReLU)
        ↓
Dense(1, lineal)
        ↓
PM2.5 estimado
```

En forma compacta:

```text
Variables meteorológicas → Preprocesamiento → MLP entrenado → PM2.5 estimado
```

---

## Arquitectura del modelo

El modelo final es un **MLP feedforward para regresión**:

```text
7 variables originales
        ↓
StandardScaler + OneHotEncoder
        ↓
10 características
        ↓
Dense(64, ReLU)
        ↓
Dense(32, ReLU)
        ↓
Dense(1, lineal)
```

### Configuración principal

| Elemento | Configuración |
|---|---|
| Framework | TensorFlow / Keras |
| Tipo de modelo | MLP feedforward |
| Problema | Regresión supervisada |
| Capa oculta 1 | 64 neuronas, ReLU |
| Capa oculta 2 | 32 neuronas, ReLU |
| Salida | 1 neurona, lineal |
| Función de pérdida | MSE |
| Optimizador | Adam |
| Learning rate inicial | 0.001 |
| Batch size | 64 |
| Máximo de épocas | 200 |
| Early stopping | Sí |
| Semilla | 42 |

---

## Dataset

Se utiliza **Beijing PM2.5**, disponible en el UCI Machine Learning Repository.

**Referencia oficial:**

> Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49

### Resumen

| Elemento | Valor |
|---|---:|
| Registros originales | 43,824 |
| Registros sin PM2.5 | 2,067 |
| Registros utilizados | 41,757 |
| Periodo | 2010–2014 |
| Frecuencia | Horaria |
| Variable objetivo | `pm2.5` |
| Licencia del dataset | CC BY 4.0 |

Los registros sin PM2.5 se excluyen del aprendizaje supervisado porque no contienen la respuesta objetivo.

```text
43,824 originales
      ↓
- 2,067 sin PM2.5
      ↓
41,757 utilizables
```

---

## Variables utilizadas

| Variable | Rol | Descripción |
|---|---|---|
| `TEMP` | Predictor | Temperatura del aire |
| `PRES` | Predictor | Presión atmosférica |
| `DEWP` | Predictor | Punto de rocío |
| `cbwd` | Predictor | Dirección combinada del viento |
| `Iws` | Predictor | Velocidad acumulada del viento |
| `Is` | Predictor | Horas acumuladas de nieve |
| `Ir` | Predictor | Horas acumuladas de lluvia |
| `pm2.5` | Objetivo | Concentración observada de PM2.5 |

---

## Preprocesamiento

La partición se realiza **cronológicamente y sin barajar**, para evitar fuga de información:

| Partición | Porcentaje | Registros |
|---|---:|---:|
| Entrenamiento | 70 % | 29,229 |
| Validación | 15 % | 6,264 |
| Prueba | 15 % | 6,264 |

Transformaciones:

- variables numéricas → `StandardScaler()`
- dirección del viento → **One-Hot Encoding** con `OneHotEncoder()`

Como resultado:

```text
7 variables originales → 10 características numéricas
```

---

## Línea base

Antes de entrenar el MLP se construyó una **regresión lineal** como modelo de referencia.

La finalidad de la línea base es comprobar si la red neuronal aporta una mejora real y medible.

---

## Resultados finales

### Conjunto de prueba reservado

| Métrica | Regresión lineal | MLP | Mejor modelo |
|---|---:|---:|---|
| MAE | 52.284 | **47.782** | MLP |
| RMSE | 70.052 | **64.584** | MLP |
| R² | 0.216 | **0.333** | MLP |

**Resultado:** el MLP supera la línea base en **3 de 3 métricas**.

### Interpretación

- **MAE**: error absoluto medio. Menor es mejor.
- **RMSE**: penaliza más los errores grandes. Menor es mejor.
- **R²**: proporción de variabilidad explicada. Mayor es mejor.

> El modelo mejora la línea base, aunque un R² de 0.333 indica que todavía existe variabilidad del PM2.5 que no está explicada por las variables disponibles.

---

## Fundamento del cálculo

Después del preprocesamiento, el modelo recibe un vector numérico `x'`.

La red puede resumirse como:

\[
\widehat{PM2.5}
=
W_3\,ReLU\left(
W_2\,ReLU\left(
W_1x'+b_1
\right)+b_2
\right)+b_3
\]

Los pesos `W` y sesgos `b` fueron **aprendidos durante el entrenamiento**.

AerQualitas no utiliza una ecuación meteorológica fija definida manualmente; aprende empíricamente la relación entre las variables meteorológicas y PM2.5.

---

## Funciones esenciales del proyecto

```text
chronological_split()
build_preprocessor()
build_linear_baseline()
build_mlp()
fit_mlp()
run_final_evaluation()
validate_user_input()
load_final_inference_assets()
predict_pm25()
```

Flujo de inferencia:

```text
validate_user_input()
        ↓
preprocessor.transform()
        ↓
model.predict()
        ↓
predict_pm25()
```

---

## Funcionalidades de la app

La interfaz contiene:

- **Inicio** — resumen ejecutivo y estimación rápida;
- **Dataset** — origen, calidad, credenciales, explorador paginado y diccionario;
- **Preprocesamiento** — división temporal, transformaciones y baseline;
- **Modelo DL** — arquitectura, entrenamiento y validación;
- **Evaluación** — métricas finales y análisis de errores;
- **Predicción** — entrada manual y ejemplos preestablecidos;
- **Ayuda** — funcionamiento, dataset, modelo y glosario;
- **Institucional** — identificación académica y técnica.

---

# Inicio rápido

## Opción A — Ejecutar la app localmente

### 1. Clonar el repositorio

```powershell
git clone https://github.com/edtech-mx-ve/aerqualitas.git
cd aerqualitas
```

### 2. Crear entorno virtual con Python 3.11

```powershell
py -3.11 -m venv .venv
```

### 3. Activar el entorno

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Comprueba:

```powershell
python --version
```

Debe mostrar:

```text
Python 3.11.x
```

### 4. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verificar TensorFlow

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Versión esperada:

```text
2.17.1
```

### 6. Ejecutar

```powershell
streamlit run app.py
```

Abrir:

```text
http://localhost:8501
```

---

# Pruebas

Antes de desplegar o subir cambios:

```powershell
pytest -q
```

No debe aparecer ningún `FAILED`.

También:

```powershell
python -m compileall src app.py
```

---

# Reproducir el pipeline completo

> Para usar la aplicación normalmente **no es necesario volver a entrenar el modelo**.

Solo ejecuta esta secuencia si deseas reproducir académicamente todo el pipeline:

```powershell
python prepare_data.py --overwrite
python train_baseline.py --overwrite
python train_mlp.py --overwrite
python evaluate_final.py
python prepare_inference.py --overwrite
pytest -q
streamlit run app.py
```

---

# Estructura del proyecto

```text
aerqualitas/
│
├── app.py
├── prepare_data.py
├── train_baseline.py
├── train_mlp.py
├── evaluate_final.py
├── prepare_inference.py
│
├── src/
│   └── aerqualitas/
│       ├── data/
│       ├── modeling/
│       ├── inference/
│       ├── config.py
│       ├── paths.py
│       ├── logging_config.py
│       └── validation.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── model/
│   ├── aerqualitas_mlp.keras
│   ├── preprocessor.joblib
│   ├── baseline_linear.joblib
│   ├── final_model_manifest.json
│   └── inference_metadata.json
│
├── assets/
│   └── logo.png
│
├── tests/
├── docs/
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

# GitHub — primera publicación

Repositorio:

<https://github.com/edtech-mx-ve/aerqualitas>

Desde la carpeta local del proyecto:

```powershell
git init
git branch -M main
git remote add origin https://github.com/edtech-mx-ve/aerqualitas.git
git status
git add .
git commit -m "Initial release AerQualitas v0.8.0"
git push -u origin main
```

Si `origin` ya existe:

```powershell
git remote -v
```

No vuelvas a agregarlo.

---

# Actualizar GitHub

Después de hacer cambios y probar:

```powershell
pytest -q
git status
git add .
git commit -m "Describe el cambio realizado"
git push
```

---

# Despliegue en Streamlit Community Cloud

**Estado:** ✅ Desplegada públicamente

**URL de producción:** [https://aerqualitas.streamlit.app/](https://aerqualitas.streamlit.app/)

El despliegue utiliza el repositorio `edtech-mx-ve/aerqualitas`, la rama `main`, `app.py` como archivo principal y Python 3.11.


## Requisitos previos

Antes de desplegar:

```powershell
pytest -q
streamlit run app.py
```

Comprueba que todas las secciones funcionen.

## Pasos

1. Ir a: <https://share.streamlit.io/>
2. Iniciar sesión con GitHub.
3. Seleccionar **Create app**.
4. Repositorio:

```text
edtech-mx-ve/aerqualitas
```

5. Rama:

```text
main
```

6. Archivo principal:

```text
app.py
```

7. En **Advanced settings**, seleccionar **Python 3.11**.
8. No agregar secretos: AerQualitas no requiere claves privadas.
9. Pulsar **Deploy**.

---

# Qué debe estar en GitHub

Debe subirse:

```text
app.py
requirements.txt
.streamlit/config.toml
src/
model/
data/processed/
assets/
README.md
```

Para reproducibilidad académica también puede conservarse:

```text
data/raw/
docs/
tests/
```

No subir:

```text
.venv/
__pycache__/
.pytest_cache/
logs/
*.pyc
tokens
contraseñas
secretos
```

---

# Solución de problemas

### PowerShell bloquea el entorno virtual

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `streamlit` no se reconoce

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### TensorFlow no carga

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Si falla:

```powershell
pip install --force-reinstall tensorflow==2.17.1
```

### Falta el modelo

```powershell
Get-ChildItem .\model
```

Deben existir al menos:

```text
aerqualitas_mlp.keras
preprocessor.joblib
final_model_manifest.json
inference_metadata.json
```

### Falta el dataset limpio

```powershell
python prepare_data.py --overwrite
```

### Puerto 8501 ocupado

```powershell
streamlit run app.py --server.port 8502
```

---

# Seguridad y robustez

AerQualitas incorpora:

- validación de tipos y rangos;
- restricciones del dominio de inferencia;
- separación temporal;
- prevención de fuga de datos;
- persistencia del modelo y preprocesador;
- verificación SHA-256 de artefactos;
- logging;
- pruebas automatizadas;
- `.gitignore` para excluir archivos temporales y sensibles.

---

# Limitaciones

- El modelo fue entrenado con datos históricos de Beijing.
- No incorpora directamente tráfico, emisiones industriales u otras fuentes externas.
- No mide PM2.5 físicamente.
- No produce AQI oficial.
- Su uso en otras ciudades requiere nuevos datos, entrenamiento y evaluación.
- Las predicciones fuera del dominio de entrenamiento se bloquean.
- Un R² de 0.333 indica que todavía existe variabilidad no explicada.

---

# Autoría

**Antonio Nicolás Toro González**  
Maestría en Inteligencia Artificial para la Transformación Digital  
Instituto Internacional de Aguascalientes

**Tutora:** Dra. Claudia Andrea Vidales Basurto

---

# Enlaces

- **Repositorio:** https://github.com/edtech-mx-ve/aerqualitas
- **Institución:** https://www.iinternacional.edu.mx/
- **Dataset:** https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata
- **DOI:** https://doi.org/10.24432/C5JS49
- **Aplicación en línea:** https://aerqualitas.streamlit.app/
- **Streamlit Community Cloud:** https://share.streamlit.io/

---

<div align="center">

### AerQualitas

**Variables meteorológicas → Preprocesamiento → MLP entrenado → PM2.5 estimado**

Versión 0.8.0 · Aplicación desplegada en Streamlit Community Cloud

Proyecto académico de Deep Learning aplicado.

</div>
