# Brain Cancer Incidence Dashboard

Este proyecto analiza la incidencia del cáncer cerebral a nivel global utilizando datos del dataset **Cancer Incidence in Five Continents (CI5)**. A través de un pipeline de procesamiento en Python y visualizaciones interactivas en Power BI, se exploran patrones por sexo, edad, subtipo y continente.

---

## Objetivos

- Evaluar la distribución de la tasa de incidencia de cáncer cerebral.
- Analizar diferencias por género, edad y geografía.
- Identificar los subtipos más comunes y su evolución.
- Presentar los resultados en un dashboard.

--- 

## Estructura del proyecto

PROYECTO ABRIL MASTER DATA SCIENCE/
├── 📂 data/                 # Archivos CSV originales 
│   ├── data.csv
│   ├── cancer_dict.csv
│   └── id_dict.csv
├── 📂 documents/           # PDFs u otros documentos utilizados o generados
│   └── cap 1.pdf, cap 2.pdf, ...
├── 📂 notebook/            # Jupyter Notebook con el análisis exploratorio
│   └── cancer.ipynb
├── 📂 outputs/             # Archivo limpio exportado desde el pipeline
│   └── cancer_brain_cleaned.csv
├── 📂 proyecto_abril/      # Otros archivos automáticos o carpetas internas
├── 📂 scripts/             # Script de procesamiento de datos
│   └── pipeline_cancer_estructurado.py
├── 📂 visuals/             # Visualizaciones y dashboard
│   ├── dashboard.pbix
│   └── otras imágenes (.png)
├── README.md              # Explicación general del proyecto



---

## Visualización en Power BI

El archivo `dashboard.pbix` contiene las visualizaciones interactivas que permiten explorar los principales hallazgos del análisis:

-  **Evolución temporal de la incidencia**
-  **Comparaciones por sexo**
-  **Distribución por edad**
-  **Subtipos tumorales más frecuentes**
-  **Distribución geográfica por continentes**

Todas las visualizaciones están acompañadas de *insights* y conclusiones clave, organizadas en diferentes pestañas dentro del dashboard.

---

## Visualización en Streamlit
Se desarrolló una aplicación web con Streamlit para ofrecer una exploración más flexible e interactiva del dataset limpio. Incluye:

- **Histograma por grupos de edad**
- **Mapas de calor de correlaciones**
- **Comparaciones por sexo y subtipo tumoral**
- **Gráficos por evolución de la incidencia según edad y sexo**
- **Gráfico multivariable que relaciona incidencia, edad, sexo y subtipo**
- **Mapa por continentes y panel comparativo filtrable**


---

## Herramientas utilizadas
- **Python: pandas, seaborn, matplotlib, scipy, statsmodels, streamlit**
- **Power BI Desktop: visualizaciones y dashboard final**
- **Jupyter Notebook: análisis exploratorio**
- **Visual Studio Code: desarrollo del pipeline**
- **Git & GitHub: control de versiones**


