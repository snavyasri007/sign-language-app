import streamlit as st
import cv2
import mediapipe as mp
import time
from gtts import gTTS
import io

st.set_page_config(page_title="AI Sign Language Translator", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
.title {font-size:55px; text-align:center; color:#4CAF50; font-weight:bold;}
.subtitle {font-size:22px; text-align:center; color:gray;}
.box {padding:20px; border-radius:15px; background:#f5f5f5; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= AUDIO =================
audio_cache = {}

def text_to_audio(text):
    if text in audio_cache:
        return audio_cache[text]

    tts = gTTS(text=text, lang="en")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    audio_bytes = fp.getvalue()
    audio_cache[text] = audio_bytes
    return audio_bytes

# ================= HOME =================
if st.session_state.page == "home":

    st.markdown('<p class="title">🧠 AI Sign Language Translator</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Bridging Communication Between Deaf & Hearing People</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="box">📷<br><b>Real-Time Detection</b></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="box">🔊<br><b>Voice Output</b></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="box">🤖<br><b>AI Powered</b></div>', unsafe_allow_html=True)

    st.image("images/isl_alphabets.jpg", use_container_width=True)

    st.markdown("## ✋ Supported Gestures")
    st.write("""
| Gesture | Meaning |
|--------|--------|
| 👍 Thumb | OK |
| ☝ Index finger | ONE |
| ✌ Two fingers | TWO |
| 🤟 Three fingers | THREE |
| ✋ Four fingers | FOUR |
| 🖐 Five fingers | FIVE |
| ✊ Closed fist | HELLO |
""")

    # ================= INTRODUCTION (UNCHANGED) =================
    st.markdown("## 📘 Introduction")
    st.write("""
This project is an AI-based Sign Language Recognition System designed to bridge the communication gap between deaf and hearing individuals.

In everyday life, people with hearing or speech impairments face challenges while communicating with others. This system provides a real-time solution by recognizing hand gestures and converting them into understandable text and speech.

The application uses computer vision techniques to detect hand landmarks and analyze finger positions. Based on these patterns, the system identifies gestures and translates them into meaningful outputs.

This project aims to create an accessible, efficient, and user-friendly platform that enhances communication and promotes inclusivity in society.
""")

    if st.button("🚀 Start Application"):
        st.session_state.page = "app"

# ================= APP =================
elif st.session_state.page == "app":

    st.title("🧏 Sign Language Recognition System")

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1)

    # 🔥 ONLY CHANGE: added hand_label support
    def get_fingers(hand_landmarks, hand_label):
        lm = hand_landmarks.landmark
        fingers = []

        if hand_label == "Right":
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)

        fingers.append(lm[8].y < lm[6].y)
        fingers.append(lm[12].y < lm[10].y)
        fingers.append(lm[16].y < lm[14].y)
        fingers.append(lm[20].y < lm[18].y)

        return fingers

    start = st.checkbox("▶ Start Camera")
    frame_box = st.image([])
    text_box = st.empty()
    audio_box = st.empty()

    if "last_text" not in st.session_state:
        st.session_state.last_text = ""

    if "audio" not in st.session_state:
        st.session_state.audio = None

    if "stable_text" not in st.session_state:
        st.session_state.stable_text = ""
    if "count" not in st.session_state:
        st.session_state.count = 0

    if start:
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and start:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected = ""

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                    label = hand_info.classification[0].label
                    f = get_fingers(hand_lm, label)

                    if f == [True, False, False, False, False]:
                        detected = "OK"
                    elif f == [False, True, False, False, False]:
                        detected = "ONE"
                    elif f == [False, True, True, False, False]:
                        detected = "TWO"
                    elif f == [False, True, True, True, False]:
                        detected = "THREE"
                    elif f == [False, True, True, True, True]:
                        detected = "FOUR"
                    elif f == [True, True, True, True, True]:
                        detected = "FIVE"
                    elif f == [False, False, False, False, False]:
                        detected = "HELLO"

            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame_box.image(frame, channels="BGR")

            if detected == st.session_state.stable_text:
                st.session_state.count += 1
            else:
                st.session_state.count = 0
                st.session_state.stable_text = detected

            if st.session_state.count > 5 and detected != "" and detected != st.session_state.last_text:
                st.session_state.last_text = detected
                st.session_state.audio = text_to_audio(detected)

            if st.session_state.last_text:
                text_box.markdown(f"## ✋ Detected: **{st.session_state.last_text}**")
                audio_box.audio(st.session_state.audio, format="audio/mp3")

            time.sleep(0.03)

        cap.release()
        cv2.destroyAllWindows()