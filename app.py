"""
app.py
------
Streamlit dashboard for the AI-Powered Sports Quiz Generation Agent.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from src.quiz_agent import QuizAgent, SPORTS, DIFFICULTIES

load_dotenv()

st.set_page_config(page_title="Sports Quiz Generator", page_icon="🏆", layout="centered")

st.title("🏆 AI Sports Quiz Generator")
st.caption(
    "RAG-powered multiple-choice sports quizzes, grounded in a ChromaDB "
    "knowledge base and live web search."
)

# --- Sidebar controls -------------------------------------------------
with st.sidebar:
    st.header("Quiz Settings")
    sport = st.selectbox("Sport", SPORTS)
    difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1)
    num_questions = st.slider("Number of questions", min_value=4, max_value=5, value=5)
    use_web_search = st.checkbox("Use live web search for fresh facts", value=True)

    generate_clicked = st.button("🎲 Generate Quiz", use_container_width=True)
    regenerate_clicked = st.button("🔄 Regenerate", use_container_width=True)

    st.divider()
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("Set GROQ_API_KEY in your environment or a .env file.")

# --- Session state ------------------------------------------------------
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "revealed" not in st.session_state:
    st.session_state.revealed = set()

# --- Generation trigger ---------------------------------------------------
if generate_clicked or regenerate_clicked:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY -- can't call the model.")
    else:
        with st.spinner("Retrieving knowledge and generating quiz..."):
            try:
                agent = QuizAgent()
                st.session_state.quiz = agent.generate_quiz(
                    sport=sport,
                    difficulty=difficulty,
                    num_questions=num_questions,
                    use_web_search=use_web_search,
                )
                st.session_state.revealed = set()
            except Exception as e:
                st.error(f"Quiz generation failed: {e}")

# --- Render quiz ---------------------------------------------------------
quiz = st.session_state.quiz
if quiz:
    st.subheader(f"Sport: {quiz['sport']}")
    st.write(f"**Difficulty:** {quiz['difficulty']}")
    st.divider()

    for i, q in enumerate(quiz["questions"], start=1):
        st.markdown(f"**Q{i}. {q['question']}**")
        for label, text in q["options"].items():
            st.write(f"{label}. {text}")

        reveal_key = f"reveal_{i}"
        if st.button(f"Show answer (Q{i})", key=reveal_key):
            st.session_state.revealed.add(i)

        if i in st.session_state.revealed:
            correct = q["correct_answer"]
            st.success(f"Correct Answer: {correct}. {q['options'][correct]}")
            st.info(q["explanation"])
        st.divider()
else:
    st.info("Choose a sport and difficulty, then click **Generate Quiz** to get started.")
