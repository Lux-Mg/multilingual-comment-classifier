import pandas as pd

df = pd.read_csv("data/benchmark_results.csv")
print(f"Total comentarios analizados: {len(df)}")

# ──────────────────────────────────────────────
# LATENCIA
# ──────────────────────────────────────────────
print("\n=== LATENCIA ===")
print(f"Tiempo total: {df['duracion_ms'].sum()/1000:.1f} seg")
print(f"Promedio (con cold start): {df['duracion_ms'].mean():.0f} ms")
print(f"Promedio (sin cold start): {df['duracion_ms'].iloc[1:].mean():.0f} ms")
print(f"P50 (mediana): {df['duracion_ms'].median():.0f} ms")
print(f"P95: {df['duracion_ms'].quantile(0.95):.0f} ms")
print(f"Min: {df['duracion_ms'].min():.0f} ms")
print(f"Max (probablemente cold start): {df['duracion_ms'].max():.0f} ms")

# ──────────────────────────────────────────────
# ACCURACY SENTIMENT
# Solo donde XLM-R sí clasificó (xlmr_sentiment no NaN)
# ──────────────────────────────────────────────
print("\n=== SENTIMENT — Qwen vs XLM-R ===")
df_sent = df[df['xlmr_sentiment'].notna()]
acc_sentiment = (df_sent['qwen_sentiment'] == df_sent['xlmr_sentiment']).mean()
print(f"Comparables (XLM-R clasificó): {len(df_sent)}")
print(f"Accuracy: {acc_sentiment*100:.1f}%")

# ──────────────────────────────────────────────
# ACCURACY LANGUAGE
# Solo donde XLM-R sí tiene idioma definido (no 'unsupported')
# ──────────────────────────────────────────────
print("\n=== LANGUAGE — Qwen vs XLM-R ===")
df_lang = df[df['xlmr_language'] != 'unsupported']
acc_lang = (df_lang['qwen_language'] == df_lang['xlmr_language']).mean()
print(f"Comparables (XLM-R detectó idioma): {len(df_lang)}")
print(f"Accuracy: {acc_lang*100:.1f}%")

# ──────────────────────────────────────────────
# CASOS RESCATADOS
# Comentarios donde XLM-R = 'unsupported' pero Qwen sí los clasificó
# ──────────────────────────────────────────────
print("\n=== RESCATES (Qwen clasificó lo que XLM-R no pudo) ===")
unsupported = df[df['xlmr_language'] == 'unsupported']
print(f"'unsupported' en XLM-R: {len(unsupported)}")
rescatados = unsupported[unsupported['qwen_language'] != 'other']
print(f"Qwen les asignó idioma definido (en/es/ru): {len(rescatados)}")

# ──────────────────────────────────────────────
# DISTRIBUCIÓN DE IDIOMAS
# ──────────────────────────────────────────────
print("\n=== DISTRIBUCIÓN POR XLM-R ===")
print(df['xlmr_language'].value_counts())
print("\n=== DISTRIBUCIÓN POR QWEN ===")
print(df['qwen_language'].value_counts())
