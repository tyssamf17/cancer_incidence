# pipeline_cancer_actualizado.py
# Script generado automáticamente desde Jupyter Notebook

# Creamos el entorno virtual y lo activamos (en bash) esto hace que todas las librerias que descarguemos se nos guarden, el kernel lo seleccionamos tambien a partir de environment
python -m venv proyecto_abril
source proyecto_abril/Scripts/activate
 
    # Librerias
pip install pandas
pip install seaborn 
pip install scipy
pip install statsmodels



# ls "C:\Users\tyssa\OneDrive\Escritorio\Proyecto Abril Master Data Science\data.csv" Para comprobar que la ruta esta bien y el archivo se encuentra ahi

# En python
import pandas as pd 
ruta = "C:/Users/tyssa/OneDrive/Escritorio/Proyecto Abril Master Data Science/data.csv"
df = pd.read_csv(ruta)
print(df.head())


filas, columnas = df.shape
print(f"El archivo tiene {filas} filas y {columnas} columnas.")
# cancer_code es el tipo de cancer (son 170 pero porque hay subtipos)

print(df.columns)
print(df['py'].head()) # Es person-year: tiempo durante el cual una población ha sido observada

#La tasa de incidencia la calcualmos con py y cases.  TI =  cases / py x 100000 (tasa de incidencia por 100 mil personas)

# Hay dos bases mas: una con info y codigos sobre los tipos de cancer (cancer_dict)
# otra con info sobre los continentes, razas...(id_dict)
# Abrimos para verlas

#Cancer_dict
ruta2 = "C:/Users/tyssa/OneDrive/Escritorio/Proyecto Abril Master Data Science/cancer_dict.csv"  
df_cancer_dict = pd.read_csv(ruta2)
print(df_cancer_dict.head())
filas, columnas = df_cancer_dict.shape
print(f"El archivo tiene {filas} filas y {columnas} columnas.")
print(df_cancer_dict.columns)


# Parece que la columna 'histo_label' tenia muchos NA por eso lo comprobamos 
na_count = df_cancer_dict['histo_label'].isna().sum()
no_na_count = df_cancer_dict['histo_label'].notna().sum()
print(f"Cantidad de NA en 'histo_label': {na_count}") #60
print(f"Cantidad de valores no NA en 'histo_label': {no_na_count}") #110

# Mostrar los valores no NA de la columna 'histo_label'
valores_no_na = df_cancer_dict['histo_label'].dropna().unique()
print("Valores no NA en 'histo_label':", valores_no_na) # Son los subtipos


# id_dict
ruta3= "C:/Users/tyssa/OneDrive/Escritorio/Proyecto Abril Master Data Science/id_dict.csv"
df_id_dict = pd.read_csv(ruta3)
print(df_id_dict.head())

filas, columnas = df_id_dict.shape
print(f"El archivo tiene {filas} filas y {columnas} columnas.")
print(df_id_dict.columns) #Parece que id_code y registry_code son las mismas 

# Fusion de las tres bases
# Tienen columnas en comun asi que vamos a unirlo con merge

# Podemos unir df con df_cancer_dict a traves de la columna cancer_code
df_merged = pd.merge(df, df_cancer_dict, on="cancer_code", how="left")

# Ahora lo unimos con id_dict a traves de la columna id_code
df_final = pd.merge(df_merged, df_id_dict, on="id_code", how="left")

# Nuestra base toda unida es df_final
print(df_final.head())
filas, columnas = df_final.shape
print(f"El archivo tiene {filas} filas y {columnas} columnas.")

# Hay muchos datos, a mi los que mas me interesan son los canceres de cerebro
# Cerebro, sistema nervioso central (C71-72)
#    Tumores astrociticos
#    Tumores oligodendrogliales y gliomas mixtos
#    Tumores ependimarios
#    Gliomas, otros
#    Meduloblastoma
#    Otros tumores embrionarios
#    Otros tumores neuroepiteliales
#    Otra morfologia especificada
#    Morfologia no especificada

print(df_final['cancer_label'].head())

# df_brain para quedarnos solo con las filas del cancer de cerebro
df_brain = df_final[df_final['cancer_label'].str.contains("brain", case=False, na=False)]
print(df_brain.head())
filas, columnas = df_brain.shape
print(f"El archivo tiene {filas} filas y {columnas} columnas.") # 1 366 860

# Si cases = 0, si hay registro pero no esta verificado
num_filas_cero = (df_brain['cases'] == 0).sum()
print("Número de filas con cases = 0:", num_filas_cero) # 1 045 134


# Eliminar filas donde 'cases' sea 0
df_brain = df_brain[df_brain['cases'] != 0]

# Cuantas filas quedan
print("Número de filas después de eliminar:", len(df_brain)) # Si cuadra: 321 726

print(df_brain['cases'].isna().sum()) # No hay NA en 'cases' 

# Vamos a ver si hay columnas duplicadas
print(df_brain.columns)

print(df_brain[['registry_code', 'id_code']]) # Es lo mismo solo que en registry_code se borran los dos ultimos numeros 
print(df_brain[['id_code', 'country_code','registry_code', 'CI5_continent', 'id_label']])
# Eliminamos registry_code 
df_brain = df_brain.drop(columns=['registry_code'])


# ¿Entre que edades estan comprendidos estas enfermedades? De 1 a 19 años 
# no es así, significa que hay 18 grupos de edad, hay una tabla don de se ponen las agrupaciones
print(sorted(df_brain['age'].unique().tolist()))

# ¿Entre que años se recogieron estos datos? 1953-2017
print(df_brain['year'])
print(sorted(df_brain['year'].unique().tolist()))

# Aproximaciones visuales a ver si nos quitamos algun dato
# Histogramas o boxplot para ver valores atipicos

import matplotlib.pyplot as plt

import seaborn as sns

# Histograma para la columna 'age'
plt.figure(figsize=(8,6))
plt.hist(df_brain['age'], bins=range(1, 20), align='left', color='skyblue', edgecolor='black')
plt.title('Distribución de grupos de edad')
plt.xlabel('Grupo de edad')
plt.ylabel('Frecuencia')
plt.xticks(range(1, 19))  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Gráfico de barras para 'sex'
plt.figure(figsize=(8,6))
sns.countplot(x='sex', data=df_brain, palette='viridis')
plt.title('Distribución de Sexos')
plt.xlabel('Sexo')
plt.ylabel('Cantidad')
plt.show()


#Para no tener que correr todo el codigo de nuevo: 
pip install pyarrow
pip install fastparquet

df_brain.to_parquet("C:/Users/tyssa/Downloads/df_brain.parquet")
df_brain = pd.read_parquet("C:/Users/tyssa/Downloads/df_brain.parquet")



# Comprobar la columna py antes de empezar 
print(df_brain['py'].isna().sum())  # Ver cuántos valores nulos hay en 'py' 318
print((df_brain['py'] == 0).sum())  # Ver cuántos ceros hay en 'py' 1362


# Calculamos la TASA DE INCIDENCIA

df_brain['incidence_rate'] = (df_brain['cases'] / df_brain['py']) * 100000
print(df_brain.head())  
print(df_brain.columns) 

#Colocar la nueva columna despues de 'py'
columnas_ordenadas = df_brain.columns.tolist()
columnas_ordenadas.remove('incidence_rate') # Hay que borrarla porque sino se duplica no se mueve de lugar
posicion = columnas_ordenadas.index('py') + 1
columnas_ordenadas.insert(posicion, 'incidence_rate')
df_brain = df_brain[columnas_ordenadas]
print(df_brain['incidence_rate'].head())  

df_brain = df_brain.loc[:, ~df_brain.columns.duplicated()]  # Elimina columnas duplicadas


# Group by 'year' para ver la evolucion del cancer a lo largo del tiempo
# No salen todos los años por los valores nulos en 'py'
df_evolucion = df_brain.groupby('year')['incidence_rate'].mean().reset_index()
print(df_evolucion.head())


import matplotlib.pyplot as plt
import seaborn as sns

# En este no salen todos los años por los valores nulos
plt.figure(figsize=(10,6))
sns.lineplot(x=df_evolucion['year'], y=df_evolucion['incidence_rate'], marker='o', color='b')
plt.title('Evolución de la Tasa de Incidencia del Cáncer de Cerebro')
plt.xlabel('Año')
plt.ylabel('Tasa de Incidencia (por 100,000)')
plt.grid(True)
plt.show()


# Vamos a hacer lo mismo pero eliminando los 0 y los NA de 'py'
df_brain_cleaned = df_brain[(df_brain['py'] > 0) & df_brain['py'].notna()]

df_evolucion = df_brain_cleaned.groupby('year').apply(
    lambda x: (x['cases'].sum() / x['py'].sum()) * 100000
).reset_index(name='incidence_rate')

# Asegurarse de que todos los años estén representados (incluso si no hay datos para un año)
all_years = pd.DataFrame({'year': range(df_brain['year'].min(), df_brain['year'].max() + 1)})
df_evolucion_completa = pd.merge(all_years, df_evolucion, on='year', how='left')

# Rellenar NaN con 0 si no hay datos para un año
df_evolucion_completa['incidence_rate'].fillna(0, inplace=True)


plt.figure(figsize=(10,6))
sns.lineplot(x=df_evolucion_completa['year'], y=df_evolucion_completa['incidence_rate'], marker='o', color='b')
plt.title('Evolución de la Tasa de Incidencia del Cáncer de Cerebro')
plt.xlabel('Año')
plt.ylabel('Tasa de Incidencia (por 100,000)')
plt.grid(True)
plt.show()


print(df_evolucion_completa) 

# Nuevo df con los 0 y NA quitados de 'py'

df_brain_cleaned = df_brain[(df_brain['py'] > 0) & df_brain['py'].notna()]
df_brain_cleaned['incidence_rate'] = (df_brain_cleaned['cases'] / df_brain_cleaned['py']) * 100000
df_brain2 = df_brain_cleaned.copy()

print(df_brain2.head())


# Agrupar por 'age' y calcular la media de la tasa de incidencia (incidence_rate)
grouped_by_age = df_brain2.groupby('age').agg(
    incidence_rate_mean=('incidence_rate', 'mean'),
    cases_sum=('cases', 'sum'),
    py_sum=('py', 'sum')
).reset_index()

print(grouped_by_age.head())

plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped_by_age, x='age', y='incidence_rate_mean', marker='o', color='b')
plt.title('Evolución de la tasa de incidencia por edad', fontsize=14)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Tasa de Incidencia Media (por 100,000 personas)', fontsize=12)
plt.xticks(range(1, 19))  
plt.show()



# Vamos a ver la tasa de incidencia por sexo
df_sex_grouped = df_brain2.groupby('sex').agg({
    'incidence_rate': 'mean'
}).reset_index()

print(df_sex_grouped) # 1 con incidencia de  6.633787 y 2 con 5.009792


plt.figure(figsize=(8,6))
sns.barplot(x='sex', y='incidence_rate', data=df_sex_grouped, palette='Set2')

plt.title('Comparación de la tasa de incidencia por sexo')
plt.xlabel('Sexo')
plt.ylabel('Tasa de incidencia por 100,000 personas')
plt.show()


# Distribucion geografica
# Cargar un archivo GeoJSON con los límites de los países
pip install folium
pip install geopandas



import folium
import geopandas as gpd

import folium
from folium import Choropleth

# Cargar el archivo GeoJSON de los países del mundo
url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/world-countries.geojson"
world_geo = gpd.read_file(url)


# Estadisticas de tendencia central para la tasa de incidencia
import scipy.stats as stats

# Media: 5.86
media = df_brain2['incidence_rate'].mean() 

# Mediana: 2.46
mediana = df_brain2['incidence_rate'].median() 

# Moda: 1.35 (frecuencia: 19)
moda = stats.mode(df_brain2['incidence_rate'], keepdims=True)  # keepdims=True evita warning

print(f"Media: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Moda: {moda.mode[0]:.2f} (frecuencia: {moda.count[0]})") 


# Medidas de dispersión: rango, varianza, desviación estándar, percentiles
# Rango (máximo - mínimo): 877.18
rango = df_brain2['incidence_rate'].max() - df_brain2['incidence_rate'].min()

# Varianza: 86.98
varianza = df_brain2['incidence_rate'].var()

# Desviación estándar: 9.33
desviacion = df_brain2['incidence_rate'].std()

# Percentiles
percentiles = df_brain2['incidence_rate'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

print(f"Rango: {rango:.2f}")
print(f"Varianza: {varianza:.2f}")
print(f"Desviación estándar: {desviacion:.2f}")
print(f"Percentiles: {percentiles}")
# Percentiles: 
# 0.25     0.773202
# 0.50     2.463691
# 0.75     7.260752
# 0.90    15.830856
# 0.95    22.489213
# 0.99    39.525692


# Distribuciones
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(16, 4))

# Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(x=df_brain2['incidence_rate'], color='lightgreen')
plt.title('Boxplot de la tasa de incidencia')
plt.xlim(0, 50)

plt.tight_layout()
plt.show()


# KDE
plt.subplot(1, 3, 3)
sns.kdeplot(df_brain2['incidence_rate'], color='purple', fill=True)
plt.title('KDE: Distribución suavizada')
plt.xlim(0, 50)  # Limitar el eje X de 0 a 50

plt.tight_layout()
plt.show()


# Correlaciones entre variables numericas
# Selección de columnas numéricas
numeric_df = df_brain2[['cases', 'py', 'incidence_rate', 'age', 'year']]

# Matriz de correlación
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación entre variables numéricas')
plt.show()




# Contraste de hipotesis: diferencias de la tasa de incidencia entre sexos
from scipy.stats import ttest_ind

# Separar los datos por sexo
hombres = df_brain2[df_brain2['sex'] == 1]['incidence_rate']
mujeres = df_brain2[df_brain2['sex'] == 2]['incidence_rate']

# Prueba t de Student
t_stat, p_value = ttest_ind(hombres, mujeres, equal_var=False) 

print(f"T-statistic: {t_stat:.2f}") # 49.94
print(f"P-value: {p_value:.4f}") # 0



# Modelo de regresion lineal
import statsmodels.api as sm

# Variables independientes (predictoras)
X = df_brain2[['age', 'sex', 'year']]
X = sm.add_constant(X)  # añade intercepto

# Variable dependiente
y = df_brain2['incidence_rate']

model = sm.OLS(y, X).fit()
print(model.summary())



# Guardamos el csv
df_brain2.to_csv("cancer_brain_cleaned.csv", index=False)
df_brain2.to_csv("cancer_brain_cleaned.csv", index=False, decimal='.')
