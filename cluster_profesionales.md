# Análisis de Clustering – Segmentación Social

## 1. Objetivo

El objetivo de este análisis es identificar **perfiles o grupos homogéneos** dentro de la base de datos utilizando técnicas de *clustering no supervisado*.  
Se busca explorar patrones latentes que permitan caracterizar distintos tipos de sujetos según sus variables sociales.

---

## 2. Metodología

Se aplicó el algoritmo **K-Means**, siguiendo los pasos clásicos:

1. Estandarización de variables
2. Determinación del número óptimo de clusters (método del codo)
3. Ajuste del modelo K-Means
4. Reducción de dimensionalidad mediante PCA
5. Interpretación de resultados

---

## 3. Script completo en Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# 1. Cargar datos
# -----------------------------
df = pd.read_csv("data.csv")

# Selección de variables numéricas
X = df.select_dtypes(include=[np.number])

# -----------------------------
# 2. Estandarización
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Método del codo
# -----------------------------
inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Modelo final K-Means
# -----------------------------
kmeans_final = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans_final.fit_predict(X_scaled)
df["cluster"] = clusters

# -----------------------------
# 5. PCA para visualización
# -----------------------------
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]

# -----------------------------
# 6. Visualización PCA
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    df["PC1"],
    df["PC2"],
    df["PC3"],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.7
)

ax.set_title("Clusters visualizados con PCA (3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Resumen descriptivo por cluster
# -----------------------------
cluster_summary = df.groupby("cluster")[X.columns].mean()
print(cluster_summary)
```
# Resultados
```python
Primeras filas del archivo:
     ID           Nombre       Sexo    Centro        Cargo   p1   p2   p3  \
0  1189    Profesional 4   Femenino  Centro 1  Categoría 1  0.6  0.6  0.8   
1  1890   Profesional 97  Masculino  Centro 2  Categoría 1  0.8  0.6  0.8   
2  1440  Profesional 102   Femenino  Centro 3  Categoría 1  0.4  0.6  0.6   
3  1344   Profesional 27   Femenino  Centro 1  Categoría 2  0.8  1.0  0.6   
4  1966   Profesional 31   Femenino  Centro 1  Categoría 2  0.8  0.8  0.8   

    p4   p5   p6   p7   p8   p9  p10  p11  
0  0.6  0.8  0.6  0.6  0.6  0.6  0.6  0.6  
1  0.6  0.6  0.6  0.6  0.8  0.6  1.0  0.8  
2  0.6  0.4  0.8  0.8  0.8  0.6  0.8  0.8  
3  1.0  1.0  0.6  0.8  1.0  0.8  0.6  0.8  
4  0.8  0.8  0.6  0.8  0.8  0.6  0.8  0.8  

Columnas numéricas que se usarán para clustering:
Index(['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11'], dtype='object')



Promedio de cada ítem por cluster:
               p1        p2        p3        p4        p5        p6        p7  \
Cluster                                                                         
0        0.774468  0.753191  0.689362  0.697872  0.791489  0.676596  0.761702   
1        0.939130  0.860870  0.808696  0.834783  0.843478  0.826087  0.878261   
2        0.630303  0.600000  0.539394  0.587879  0.630303  0.593939  0.703030   

               p8        p9       p10       p11  
Cluster                                          
0        0.782979  0.663830  0.685106  0.778723  
1        0.895652  0.808696  0.939130  0.886957  
2        0.660606  0.587879  0.600000  0.666667  

Cantidad de personas por cluster:
Cluster
0    47
2    33
1    23
Name: count, dtype: int64


Archivo guardado: /CLUSTER_JJ_con_clusters.xlsx
```

# Gráficos

# Método del codo
![Método del codo](https://raw.githubusercontent.com/mbburon/data-portfolio-social-analytics/bed834b0a8a03f36b7d4c4f7cb3743024861de4d/IMAGENES/metodo_codo_c1.png)

# Dendrograma
![Dendrograma](https://raw.githubusercontent.com/mbburon/data-portfolio-social-analytics/bed834b0a8a03f36b7d4c4f7cb3743024861de4d/IMAGENES/dendrograma_c1.png)

# Clusters
![Cluster3D](https://raw.githubusercontent.com/mbburon/data-portfolio-social-analytics/bed834b0a8a03f36b7d4c4f7cb3743024861de4d/IMAGENES/clusters_3D_c1.png)


