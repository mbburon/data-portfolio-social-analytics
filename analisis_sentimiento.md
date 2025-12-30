# Análisis de Sentimiento


# Contexto del problema

Este análisis surge de la necesidad de pesquisar necesidades formativas en un grupo de profesionales que se desempeñan en una institución educativa

Se dispone de las respuestas a una pregunta abierta de un cuestionario referida a necesidades formativas, a la cual responden 120 personas variando la extensión de la respuesta entre los 0 y 550 caracteres. 

Las alternativas de análisis son

Trabajar desde el Paraigma Interpretativo de las Ciencias Sociales y categorizar las respuestas según temas emergentes realizando posteriormente un análisis temático transversal para interpretar el conjunto de respuestas

Otra alternativa, según los objetivos de la pregunta realizada, es revisar las palabras cuantitativamente según su frecuencia de aparición, con el fin de pesquisar temas recurrentes que reflejen los intereses de los consultados. A la vez, la connotación emotiva de las respuestas puede ser obtenida mediante un análisis de sentimiento.

Finalmente, se optó por el análisis cuantitativo de frecuencias y el análisis de sentimiento

# Porcedimiento a realizar

Análisis de texto en español:
- Limpieza y lematización con spaCy
- Nube de palabras (WordCloud)
- Análisis básico de sentimiento

Requisitos:
- pandas
- matplotlib
- wordcloud
- spacy
- textblob
- openpyxl


# Importación de librerías

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
# Interpretación de los resultados, nube de palabras y análisis de sentimiento

El análisis cualitativo de los resultados, permite caracterizar el discurso de los participantes como principalmente técnico, descriptivo y orientado a la mejora de la intervención profesional

La nube de palabras evidencia una alta frecuencia de términos como intervención, capacitación, trabajo, formación, proceso, estrategia, conocimiento, equipo y contexto, lo que indica que los comentarios se centran en necesidades formativas concretas, fortalecimiento de competencias y optimización de los procesos de intervención. La presencia de conceptos asociados a trayectoria, rol profesional, evaluación e interculturalidad sugiere además una reflexión situada, vinculada tanto a la práctica cotidiana como a la complejidad del contexto de intervención.

Estos hallazgos se ven reforzados por el análisis de sentimiento, que muestra un predominio claro de comentarios neutrales (87,4%), junto con una proporción menor de comentarios positivos (10,7%) y una presencia marginal de comentarios negativos (1,9%). Esta distribución indica que el espacio de respuesta es utilizado mayoritariamente para formular observaciones técnicas, diagnósticos y propuestas, más que para expresar evaluaciones emocionales o juicios críticos.

En conjunto, ambos análisis sugieren que los participantes adoptan una posición reflexiva y profesional, orientada a identificar brechas de capacitación y oportunidades de mejora, dentro de un clima discursivo más bien constructivo. La baja carga negativa y la presencia de términos asociados al aprendizaje y la mejora continua refuerzan la pertinencia de utilizar estos insumos como base para diseñar estrategias formativas diferenciadas y ajustadas a las necesidades reales del equipo
