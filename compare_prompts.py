import pandas as pd

# ── Cargar ambos benchmarks ──
df_es = pd.read_csv("data/benchmark_results_es.csv")
df_en = pd.read_csv("data/benchmark_results_en.csv")

print(f"Comentarios analizados: {len(df_es)} (ES) y {len(df_en)} (EN)")
assert len(df_es) == len(df_en), "Diferentes tamaños de muestra"


# ──────────────────────────────────────────────
# 1. ACCURACY vs XLM-R (ambos prompts)
# ──────────────────────────────────────────────
print("\n=== ACCURACY vs XLM-R ===")

# Sentiment
df_es_sent = df_es[df_es['xlmr_sentiment'].notna()]
df_en_sent = df_en[df_en['xlmr_sentiment'].notna()]
# TU TAREA: calcular acc_sent_es y acc_sent_en (comparando qwen_sentiment vs xlmr_sentiment)
acc_sent_es = (df_es_sent['qwen_sentiment'] == df_es_sent['xlmr_sentiment']).mean()
acc_sent_en = (df_en_sent['qwen_sentiment'] == df_en_sent['xlmr_sentiment']).mean()
print(f"Sentiment — ES prompt: {acc_sent_es*100:.1f}%")
print(f"Sentiment — EN prompt: {acc_sent_en*100:.1f}%")
print(f"Diferencia: {(acc_sent_en - acc_sent_es)*100:+.1f} puntos")

# Language
df_es_lang = df_es[df_es['xlmr_language'] != 'unsupported']
df_en_lang = df_en[df_en['xlmr_language'] != 'unsupported']
# TU TAREA: calcular acc_lang_es y acc_lang_en
acc_lang_es = (df_es_lang['qwen_language'] == df_es_lang['xlmr_language']).mean()   
acc_lang_en = (df_en_lang['qwen_language'] == df_en_lang['xlmr_language']).mean()
print(f"\nLanguage — ES prompt: {acc_lang_es*100:.1f}%")
print(f"Language — EN prompt: {acc_lang_en*100:.1f}%")
print(f"Diferencia: {(acc_lang_en - acc_lang_es)*100:+.1f} puntos")


# ──────────────────────────────────────────────
# 2. DISTRIBUCIONES DE LANGUAGE
# ──────────────────────────────────────────────
print("\n=== DISTRIBUCIÓN DE IDIOMA ===")
print("Prompt ES detectó:")
print(df_es['qwen_language'].value_counts())
print("\nPrompt EN detectó:")
print(df_en['qwen_language'].value_counts())


# ──────────────────────────────────────────────
# 3. CASOS DONDE ES Y EN DIFIEREN (merge)
# ──────────────────────────────────────────────
# Merge por comment_id — esto une los DataFrames por la columna común
merged = df_es[['comment_id', 'text', 'xlmr_language', 'qwen_language', 'qwen_sentiment']].merge(
    df_en[['comment_id', 'qwen_language', 'qwen_sentiment']],
    on='comment_id',
    suffixes=('_es', '_en')  # las columnas duplicadas reciben este sufijo
)

# Filtrar casos donde Qwen dio resultados distintos entre ES y EN
diff_lang = merged[merged['qwen_language_es'] != merged['qwen_language_en']]
diff_sent = merged[merged['qwen_sentiment_es'] != merged['qwen_sentiment_en']]

print(f"\n=== DESACUERDOS ENTRE PROMPT ES Y EN ===")
print(f"Language difiere: {len(diff_lang)} casos")
print(f"Sentiment difiere: {len(diff_sent)} casos")

# Mostrar los casos de diferencia en language
print(f"\n--- Casos donde el idioma detectado cambió: ---")
for _, row in diff_lang.iterrows():
    print(f"[XLM-R={row['xlmr_language']}] {row['text'][:70]}...")
    print(f"    ES prompt: {row['qwen_language_es']}  |  EN prompt: {row['qwen_language_en']}")
    print()
