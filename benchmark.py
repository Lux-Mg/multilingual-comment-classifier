import pandas as pd
import time
from clasificar import clasificar_comentario

# ── Configuración ──
N_COMENTARIOS = 100
PROMPT_LANG = 'en'  # 'es' o 'en' — para experimento A/B
INPUT_CSV = "data/comentarios.csv"
OUTPUT_CSV = f"data/benchmark_results_{PROMPT_LANG}.csv"

# ── Cargar y muestrear ──
df = pd.read_csv(INPUT_CSV)
muestra = df.sample(n=N_COMENTARIOS, random_state=42)
print(f"Procesando {N_COMENTARIOS} comentarios con Qwen 2.5 7B...")
print()

# ── Procesar y medir ──
resultados = []
inicio_total = time.perf_counter()

for i, (indice, fila) in enumerate(muestra.iterrows(), start=1):
    # TU TAREA: medir tiempo, clasificar, guardar el dict en resultados
    inicio = time.perf_counter()
    resultado = clasificar_comentario(fila['text'], prompt_lang=PROMPT_LANG)
    duracion_ms = (time.perf_counter() - inicio) * 1000
    
    # TU TAREA: armar el dict y agregarlo a resultados
    # Campos: comment_id, text, xlmr_language, xlmr_sentiment, qwen_language, qwen_sentiment, duracion_ms
    resultados.append({
        'comment_id': fila['comment_id'],
        'text': fila['text'],
        'xlmr_language': fila['language'],
        'xlmr_sentiment': fila['xlmr_sentiment'],
        'qwen_language': resultado['language'],
        'qwen_sentiment': resultado['sentiment'],
        'duracion_ms': round(duracion_ms, 2)
    })
    
    # Progreso cada 10
    if i % 10 == 0:
        print(f"  Procesados: {i}/{N_COMENTARIOS}")

tiempo_total = time.perf_counter() - inicio_total

# ── Guardar resultados ──
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(OUTPUT_CSV, index=False)

print()
print(f"Tiempo total: {tiempo_total:.1f} segundos")
print(f"Promedio: {tiempo_total/N_COMENTARIOS:.2f} seg/comentario")
print(f"Resultados guardados en: {OUTPUT_CSV}")
