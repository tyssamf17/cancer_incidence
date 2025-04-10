#STREAMLIT PARA CREAR DASHBOARD A PARTIR DE LOS GRAFICOS DEL NOTEBOOK cancer.ipynb
#streamlit run "outputs/dashboard_streamlit.py"  # en bash



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pycountry


# Configuración de página
st.set_page_config(page_title="Brain Cancer Dashboard", layout="wide")

# Título principal
st.title("📊 Dashboard - Incidencia de Cáncer Cerebral")

# Cargar datos
df = pd.read_csv("outputs/cancer_brain_cleaned.csv")

# Gráfico 1: Correlaciones
cols = ["age", "incidence_rate", "year", "py", "cases"]
df_corr = df[cols].copy()
st.subheader(" Mapa de correlaciones")
corr = df_corr.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Filtro por sexo
sex_filter = st.sidebar.multiselect("Selecciona sexo", df["sex"].unique(), default=df["sex"].unique())
df = df[df["sex"].isin(sex_filter)]

# Gráfico 2: Casos totales por edad
st.subheader("Distribución de casos por edad")
df_age = df.groupby("age")["cases"].sum().reset_index()
fig1 = px.bar(df_age, x="age", y="cases", title="Casos totales por edad")
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 3: Casos totales por sexo
st.subheader("Distribución de casos por sexo")
df_sex_cases = df.groupby("sex")["cases"].sum().reset_index()
df_sex_cases["sex"] = df_sex_cases["sex"].map({1: "Male", 2: "Female"})
fig_sex = px.bar(
    df_sex_cases,
    x="sex",
    y="cases",
    color="sex",
    text_auto=True,
    title="Casos totales por sexo",
    labels={"cases": "Total de casos", "sex": "Sexo"}
)
st.plotly_chart(fig_sex, use_container_width=True)

# Gráfico 4: Casos totales por subtipo de cancer
st.subheader("Casos totales por subtipo de cáncer cerebral")
df_histo_cases = df.groupby("histo_label")["cases"].sum().reset_index()
df_histo_cases = df_histo_cases.sort_values(by="cases", ascending=False)
fig_histo = px.bar(
    df_histo_cases,
    x="cases",
    y="histo_label",
    orientation="h",
    text_auto=True,
    title="Casos totales por subtipo de tumor",
    labels={"cases": "Total de casos", "histo_label": "Subtipo"}
)

st.plotly_chart(fig_histo, use_container_width=True)



# Gráfico 5: Incidencia por sexo
st.subheader("Tasa de incidencia media por edad y sexo")
df_age_sex = df.groupby(["age", "sex"])["incidence_rate"].mean().reset_index()
df_age_sex["sex"] = df_age_sex["sex"].map({1: "Male", 2: "Female"})
fig_age_sex = px.line(
    df_age_sex,
    x="age",
    y="incidence_rate",
    color="sex",
    markers=True,
    labels={"age": "Grupo de edad", "incidence_rate": "Tasa de incidencia", "sex": "Sexo"},
    title="Evolución de la tasa de incidencia por edad según el sexo"
)

fig_age_sex.update_layout(xaxis=dict(dtick=1))
st.plotly_chart(fig_age_sex, use_container_width=True)

# Gráfico 6
st.subheader("Relación entre incidencia, edad, sexo y subtipo de tumor")

df_bubble = df.groupby(["age", "sex", "histo_label"]).agg({
    "incidence_rate": "mean",
    "cases": "sum"
}).reset_index()
df_bubble["sex"] = df_bubble["sex"].map({1: "Male", 2: "Female"})

fig_bubble = px.scatter(
    df_bubble,
    x="age",
    y="incidence_rate",
    size="cases",
    color="sex",
    hover_name="histo_label",
    labels={
        "age": "Edad",
        "incidence_rate": "Tasa de incidencia",
        "cases": "Casos",
        "sex": "Sexo"
    },
    title="Relación entre incidencia, edad, sexo y subtipo de cáncer cerebral"
)

fig_bubble.update_layout(xaxis=dict(dtick=1))
st.plotly_chart(fig_bubble, use_container_width=True)


# Gráfico 7: Evolución temporal
st.subheader("Evolución de la incidencia en el tiempo")
df_year = df.groupby(["year", "sex"])["incidence_rate"].mean().reset_index()
fig3 = px.line(df_year, x="year", y="incidence_rate", color="sex", markers=True)
st.plotly_chart(fig3, use_container_width=True)


# Gráfico 8: Tasa media de incidencia por pais/continente

# Función para convertir códigos numéricos a ISO Alpha-3
def numeric_to_alpha3(numeric_code):
    try:
        country = pycountry.countries.get(numeric=str(int(numeric_code)))
        return country.alpha_3 if country else None
    except:
        return None

# Crear nueva columna con códigos ISO
df["country_alpha3"] = df["country_code"].apply(numeric_to_alpha3)
# Agrupar por país
df_country = df.groupby("country_alpha3")["incidence_rate"].mean().reset_index()
import plotly.express as px

st.subheader("Mapa interactivo: Tasa media de incidencia por país")

fig = px.choropleth(
    df_country,
    locations="country_alpha3",
    color="incidence_rate",
    hover_name="country_alpha3",
    color_continuous_scale="YlOrRd",
    projection="natural earth"
)

st.plotly_chart(fig, use_container_width=True)



# Gráfico 9: Tasa de incidencia a lo largo del tiempo por continente
st.subheader("Evolución temporal por continente")
df_year_cont = df.groupby(["year", "CI5_continent"])["incidence_rate"].mean().reset_index()

continent_dict = {
    1: "Africa",
    2: "South America",
    3: "North America",
    4: "Asia",
    5: "Europe",
    6: "Oceania"
}
df_year_cont["continent"] = df_year_cont["CI5_continent"].map(continent_dict)

fig = px.line(df_year_cont, x="year", y="incidence_rate", color="continent",
              labels={"incidence_rate": "Tasa de incidencia"})
st.plotly_chart(fig, use_container_width=True)


# Gráfico 10: Comparador personalizado por filtros (con widgets)
st.subheader("Comparador personalizado por continente")

continent_dict = {
    1: "Africa",
    2: "South America",
    3: "North America",
    4: "Asia",
    5: "Europe",
    6: "Oceania"
}


continent_options = {v: k for k, v in continent_dict.items()}
nombre_continente = st.selectbox("Selecciona un continente", list(continent_options.keys()))

codigo_continente = continent_options[nombre_continente]
df_filtro = df[df["CI5_continent"] == codigo_continente]

st.metric("Tasa media de incidencia", f"{df_filtro['incidence_rate'].mean():.2f}")
st.metric("Total de casos", int(df_filtro["cases"].sum()))



