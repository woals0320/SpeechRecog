import io
import librosa
import soundfile as sf
import streamlit as st
import torch
from audiorecorder import audiorecorder
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoProcessor,
    Wav2Vec2ForCTC,
)
    
# 모델 경로
WAV2VEC_PATH = "my-finetuned-wav2vec2"
KOBART_PATH = "KoBART"

@st.cache_resource(show_spinner="🤖 model loading...")
def load_models():
    processor = AutoProcessor.from_pretrained(WAV2VEC_PATH)
    asr_model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_PATH)
    
    kobart_tokenizer = AutoTokenizer.from_pretrained(KOBART_PATH)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(KOBART_PATH)
    return processor, asr_model, kobart_tokenizer, gec_model

processor, asr_model, kobart_tokenizer, gec_model = load_models()

device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model.to(device)
gec_model.to(device)

st.title("🗣 Korean Transcriber for foreigner")
st.caption("When you input your voice, a corrected Korean sentence is output. (Wav2Vec2 → KoBART)")

# 좌우 레이아웃 설정
left_col, right_col = st.columns(2)

with left_col:
    st.header("1️⃣ Voice input and transcription")
    audio = audiorecorder("🎙Start Recoding", "🛑Recoding Complete")

    if len(audio) == 0:
        st.info("First, record your voice with a microphone..")
        st.stop()

    with st.spinner("⏱ audio processing..."):
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        y, sr = sf.read(wav_bytes)

        # 샘플레이트 16kHz로 통일
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 모노로 변환
        if y.ndim > 1:
            y = librosa.to_mono(y.T)

        # Wav2Vec2 ASR 추론
        input_values = processor(y, sampling_rate=sr, return_tensors="pt", padding="longest").input_values.to(device)
        with torch.no_grad():
            logits = asr_model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0]

    st.text_area("📝 Transcription Result", transcription, height=200)

with right_col:
    st.header("2️⃣ Sentence Correction")
    if 'transcription' in locals():
        if st.button("✏️ Grammar Correction"):
            with st.spinner("✏️ Correcting with KoBART..."):
            
                inputs = kobart_tokenizer(transcription, return_tensors="pt", max_length=128, truncation=True).to(device)
                inputs.pop("token_type_ids", None)
                with torch.no_grad():
                    generated_ids = gec_model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                    )
                corrected = kobart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # st.success("✅ Correction Complete!")
            st.text_area("🪄 Correction Result", corrected, height=200)

            st.download_button(
                label="📥 Result TXT download",
                data=corrected,
                file_name="corrected_korean.txt",
                mime="text/plain",
            )
    else:
        st.info("Please record your voice first and then transcribe it..")

st.markdown("---")
st.caption("2025 speech recognition PBL project by Team Whiroro")