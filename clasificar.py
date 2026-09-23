import os
import json
from ollama import Client                # ← reemplaza `import ollama`


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Prompts (mismo contenido, distinto idioma) ──
PROMPT_ES = """Analiza el comentario de película dentro de las etiquetas <comentario>.
Clasifica su sentimiento (positive/negative/neutral) y detecta el idioma EN QUE ESTÁ ESCRITO EL COMENTARIO.

<comentario>
{comentario}
</comentario>"""

PROMPT_EN = """Analyze the movie comment within the <comment> tags.
Classify its sentiment (positive/negative/neutral) and detect the language IT IS WRITTEN IN.

<comment>
{comentario}
</comment>"""

client = Client(host=OLLAMA_HOST)

def clasificar_comentario(comentario, prompt_lang='en'):
    # Elegir prompt según el idioma
    if prompt_lang == 'en':
        prompt = PROMPT_EN.format(comentario=comentario)
    else:
        prompt = PROMPT_ES.format(comentario=comentario)
    
    response = client.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        format={
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "language": {"type": "string", "enum": ["en", "es", "ru", "other"]}
            },
            "required": ["sentiment", "language"]
        }
    )
    
    return json.loads(response["message"]["content"])


if __name__ == "__main__":
    comentario_test = "This movie was a complete waste of time, terrible acting"
    
    print("=== Con prompt ES ===")
    print(clasificar_comentario(comentario_test, prompt_lang='es'))

    print("\n=== Con prompt EN (default) ===")
    print(clasificar_comentario(comentario_test))
