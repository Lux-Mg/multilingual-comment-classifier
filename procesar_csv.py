import pandas as pd
from clasificar import clasificar_comentario

# ── Leer el CSV ──
# TU TAREA: leer data/comentarios.csv con pandas
df = pd.read_csv("data/comentarios.csv")

# ── Explorar qué cargamos ──
print("Total de filas:", len(df))
print("Columnas:", df.columns.tolist())
print()
print("Primeras 3 filas:")
print(df.head(3))

muestra = df.sample(10, random_state=42)

print()
print("Muestra de 10 comentarios:")
for indice, fila in muestra.iterrows():
    print(f"- [{fila['language']}] {fila['text'][:80]}")
    
    
print()
print("Clasificando con Qwen")
print()

for indice, fila in muestra.iterrows():
    resultado = clasificar_comentario(fila['text'])
    
    print(f"[{indice}] {fila['text'][:60]}...")
    print(f" CSV (XLM-R): language={fila['language']}, sentiment={fila['xlmr_sentiment']}")
    print(f"  Qwen:        language={resultado['language']}, sentiment={resultado['sentiment']}")
    print()
    