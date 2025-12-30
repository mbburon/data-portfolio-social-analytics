Análisis de texto en español:
- Limpieza y lematización con spaCy
- Nube de palabras (WordCloud)
- Análisis básico de sentimiento

Autor: Martín Burón
Perfil: Sociología / Ciencia de Datos

Requisitos:
- pandas
- matplotlib
- wordcloud
- spacy
- textblob
- openpyxl


# Librerías

```python
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import spacy
from textblob import TextBlob
```

# Configuración general

```python
RUTA_ARCHIVO = "Nube de palabras Python.xlsx"
COLUMNA_TEXTO = "Espacio para comentarios adicionales"

TOP_N = 120
MIN_LEN = 3
ALLOWED_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

ARCHIVO_SALIDA = "nube_palabras.png"
```
# Carga del modelo spaCy (ES)

```python
def cargar_modelo_spacy():
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("es_core_news_sm")
        return spacy.load("es_core_news_sm")


nlp = cargar_modelo_spacy()
```
# Stopwords

```python
STOP_SPACY = {w.lower() for w in nlp.Defaults.stop_words}
STOP_WC = {w.lower() for w in STOPWORDS}

STOP_CUSTOM = {
    "si", "no", "mas", "ademas", "además", "etc",
    "hola", "gracias", "buenas", "buenos",
    "dias", "días", "tardes", "noches",
    "sr", "sra", "sres",
    "https", "www", "com"
}

STOPWORDS_TOTAL = STOP_SPACY | STOP_WC | STOP_CUSTOM
```

# Funciones de preprocesamiento

```python
def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def lematizar_y_filtrar(texto: str) -> list[str]:
    doc = nlp(normalizar_texto(texto))
    tokens = []

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.like_num or tok.like_url or tok.like_email:
            continue
        if tok.is_stop or tok.pos_ not in ALLOWED_POS:
            continue

        lemma = tok.lemma_.lower().strip()

        if len(lemma) < MIN_LEN:
            continue
        if lemma in STOPWORDS_TOTAL:
            continue

        tokens.append(lemma)

    return tokens
```

# Análisis de sentimiento

```python
def clasificar_sentimiento(texto: str) -> str:
    polaridad = TextBlob(texto).sentiment.polarity
    if polaridad > 0:
        return "Positivo"
    elif polaridad < 0:
        return "Negativo"
    return "Neutro"
```

# Pipeline principal

```python
def main():
    
    1. Cargar datos
    
    df = pd.read_excel(RUTA_ARCHIVO)

    comentarios = (
        df[COLUMNA_TEXTO]
        .astype(str)
        .apply(str.strip)
    )
    comentarios = comentarios[comentarios != ""]
```
    2. Tokenización y lematización
  
```python
    tokens_totales = []
    for texto in comentarios:
        tokens_totales.extend(lematizar_y_filtrar(texto))
```
    3. Frecuencias
    
```python
    
    frecuencias = Counter(tokens_totales)
    freq_top = dict(frecuencias.most_common(TOP_N))
```
  
    4. Nube de palabras
```python   
    wc = WordCloud(
        background_color="white",
        width=1400,
        height=800,
        collocations=False
    ).generate_from_frequencies(freq_top)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Nube de palabras (lematizada – Top {TOP_N})", fontsize=16)
    plt.show()

    wc.to_file(ARCHIVO_SALIDA)
    print(f"Nube guardada como '{ARCHIVO_SALIDA}'")
```
    
    5. Sentimiento
```python
    df["Sentimiento"] = comentarios.apply(clasificar_sentimiento)

    conteo = df["Sentimiento"].value_counts()
    porcentajes = (conteo / conteo.sum() * 100).round(1)

    print("\n Distribución de sentimientos:")
    print(conteo)

    print("\n Porcentajes:")
    print(porcentajes.astype(str) + "%")
```


# Ejecución

```python
if __name__ == "__main__":
    main()
```

# Resultados

```python
✔ Download and installation successful
You can now load the package via spacy.load('es_core_news_sm')
⚠ Restart to reload dependencies
If you are in a Jupyter or Colab notebook, you may need to restart Python in
order to load all the package's dependencies. You can do this by selecting the
'Restart kernel' or 'Restart runtime' option.
```

![Nube de palabras](https://github.com/mbburon/data-portfolio-social-analytics/blob/1bc928c34a19bd50fdf7b13e9bd8ac511273015b/IMAGENES/nube_palabras.png)


```python
Nube guardada como 'nube_palabras.png'

Distribución de sentimientos:
Sentimiento
Neutro      90
Positivo    11
Negativo     2
Name: count, dtype: int64

Porcentajes:
Sentimiento
Neutro      87.4%
Positivo    10.7%
Negativo     1.9%
Name: count, dtype: object
```
