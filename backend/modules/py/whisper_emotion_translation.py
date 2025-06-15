import sys, json
try:
    import whisper
    from transformers import pipeline
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    from aarunex_lwl_hybrid_v4 import AaruNexLWLv4
except ImportError:
    print(json.dumps({"error": "Missing dependencies."}))
    sys.exit(1)

if len(sys.argv) != 3:
    print(json.dumps({"error": "Usage: script <audio_file> <target_lang>"}))
    sys.exit(1)

audio_file = sys.argv[1]
target_lang = sys.argv[2]

try:
    model = whisper.load_model("base")
    text = model.transcribe(audio_file).get("text", "")
except Exception as e:
    print(json.dumps({"error": f"Transcription failed: {str(e)}"}))
    sys.exit(1)

try:
    tone = SentimentIntensityAnalyzer().polarity_scores(text)
    emotion = "positive" if tone["compound"] > 0.5 else "negative" if tone["compound"] < -0.5 else "neutral"
except:
    emotion = "neutral"

lwl = AaruNexLWLv4()
result = lwl.translate(text, target_lang)
result.original_text = text
result.emotional_tone = emotion
print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
