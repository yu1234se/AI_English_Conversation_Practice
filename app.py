import base64
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # これを追加

import re
import io
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
from audio_transcribe import AudioTranscriber
from kokoro import KPipeline
from conversation_agent import generate_response



def generate_audio(pipeline, text, speed=1.0):
    """音声を生成してバイト列を返す（速度調整付き）"""
    generator = pipeline(text, voice='af_heart')
    chunks = [audio for _, _, audio in generator]
    full_audio = np.concatenate(chunks, axis=0)
    
    # 速度調整
    if speed != 1.0:
        new_length = int(len(full_audio) / speed)
        indices = np.clip(np.round(np.arange(new_length) * speed), 0, len(full_audio)-1).astype(int)
        full_audio = full_audio[indices]
    
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, 24000, format='WAV')
    return buffer.getvalue()

def custom_audio_recorder():
    """カスタム音声レコーダー"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", disabled=st.session_state.recording_active,
                    use_container_width=True, key="start_recording"):
            st.session_state.recording_active = True
            st.session_state.audio_data = None
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording_active,
                    use_container_width=True, key="stop_recording"):
            st.session_state.recording_active = False
            st.rerun()
    
    # 録音処理
    if st.session_state.recording_active:
        st.info("Recording... Speak now! (Max 5 minutes)")
        fs = 16000
        seconds = 300
        
        # 録音開始
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
        st.session_state.recording = recording
        st.session_state.fs = fs
        st.session_state.recording_in_progress = True
    
    # 録音停止後の処理
    if not st.session_state.recording_active and 'recording' in st.session_state:
        recording = st.session_state.recording
        
        # 無音部分をトリミング
        audio_data = recording.squeeze()
        threshold = 0.01
        non_silent = np.where(np.abs(audio_data) > threshold)[0]
        
        if len(non_silent) > 0:
            start = max(0, non_silent[0] - 100)
            end = min(len(audio_data), non_silent[-1] + 100)
            audio_data = audio_data[start:end]
        else:
            audio_data = np.array([])
        
        if len(audio_data) > 0:
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile_path = tmpfile.name
                sf.write(tmpfile_path, audio_data, st.session_state.fs, format='WAV')
            
            # セッション状態に保存
            with open(tmpfile_path, "rb") as f:
                st.session_state.audio_data = f.read()
            st.session_state.audio_path = tmpfile_path
            
            # 音声プレビュー
            st.audio(st.session_state.audio_data, format="audio/wav")
        else:
            st.warning("No audio recorded. Please try again.")
        
        # 録音データクリア
        del st.session_state.recording
    
    return st.session_state.get('audio_data') is not None

def transcribe_audio(audio_path):
    """音声を文字起こし"""
    transcriber = AudioTranscriber()
    segments = transcriber.transcribe(audio_path)
    return " ".join(text for _, _, text in segments)

# メインアプリ
st.title("AI English Conversation Practice")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    # US Englishに固定 (lang_code='a')
    st.session_state.pipeline = KPipeline(lang_code='a')
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "audio_speed" not in st.session_state:
    st.session_state.audio_speed = 1.0

# サイドバー設定（音声速度のみ）
with st.sidebar:
    st.subheader("Settings")
    # 音声速度調整
    st.session_state.audio_speed = st.slider(
        "Speech Speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust the playback speed of AI responses"
    )

# ウェルカムメッセージ
if len(st.session_state.messages) == 0:
    st.info("Click 'Start Recording' to begin your English conversation practice!")

# 録音セクション
st.header("🎤 Record Your Voice")
if custom_audio_recorder() and st.button("Transcribe and Send"):
    with st.spinner("Transcribing your speech..."):
        transcribed_text = transcribe_audio(st.session_state.audio_path)
        
        # 一時ファイル削除
        try:
            os.unlink(st.session_state.audio_path)
        except:
            pass
        
        # メッセージ追加
        st.session_state.messages.append({
            "role": "user", 
            "content": transcribed_text,
            "type": "voice"
        })
        
        # 状態リセット
        st.session_state.audio_data = None
        st.session_state.audio_path = None

# 会話処理
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    with st.spinner("AI is thinking..."):
        # AI応答生成（分離したモジュールを使用）
        ai_response = generate_response(
            last_user_msg, 
            st.session_state.messages
        )
        
        # 音声生成（速度調整付き）
        audio_bytes = generate_audio(
            st.session_state.pipeline, 
            ai_response,
            speed=st.session_state.audio_speed
        )
        
        # メッセージ追加
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_response,
            "audio_bytes": audio_bytes
        })
        
    # 画面更新
    st.rerun()

# 会話表示
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if msg["type"] == "voice":
                st.caption("🎤 You said:")
            st.write(msg["content"])
        else:
            # 音声プレイヤー (最後のメッセージは自動再生)
            autoplay = (i == len(st.session_state.messages) - 1)
            st.audio(msg["audio_bytes"], format='audio/wav', autoplay=autoplay)
            
            # 折りたたみ式テキスト表示
            with st.expander("Click to see the English text"):
                st.write(msg["content"])
