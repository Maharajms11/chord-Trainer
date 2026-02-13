import random
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Guitar Chord Tone Trainer", layout="wide")

NOTE_LABELS_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_LABELS_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
QUIZ_NOTE_OPTIONS = sorted(
    set(
        NOTE_LABELS_SHARP
        + NOTE_LABELS_FLAT
        + ["Cb", "E#", "Fb", "B#", "C##", "F##", "G##", "D##", "A##"]
    )
)
ROOT_OPTIONS = [
    ("C", 0),
    ("C#/Db", 1),
    ("D", 2),
    ("D#/Eb", 3),
    ("E", 4),
    ("F", 5),
    ("F#/Gb", 6),
    ("G", 7),
    ("G#/Ab", 8),
    ("A", 9),
    ("A#/Bb", 10),
    ("B", 11),
]

# Standard tuning: low E to high E.
STRING_OPEN_PCS = [4, 9, 2, 7, 11, 4]
STRING_LABELS = ["6 (E)", "5 (A)", "4 (D)", "3 (G)", "2 (B)", "1 (e)"]

LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
NATURAL_PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEGREE_TO_STEP = {
    "1": 0,
    "2": 1,
    "b3": 2,
    "3": 2,
    "4": 3,
    "b5": 4,
    "5": 4,
    "#5": 4,
    "b7": 6,
    "bb7": 6,
    "7": 6,
    "9": 1,
    "#9": 1,
    "13": 5,
}
OFFSET_TO_ACCIDENTAL = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}

CHORDS: Dict[str, Dict[str, object]] = {
    "Major": {"intervals": [0, 4, 7], "degrees": ["1", "3", "5"], "group": "Common"},
    "Minor": {"intervals": [0, 3, 7], "degrees": ["1", "b3", "5"], "group": "Common"},
    "Diminished": {"intervals": [0, 3, 6], "degrees": ["1", "b3", "b5"], "group": "Common"},
    "Augmented": {"intervals": [0, 4, 8], "degrees": ["1", "3", "#5"], "group": "Common"},
    "Sus2": {"intervals": [0, 2, 7], "degrees": ["1", "2", "5"], "group": "Common"},
    "Sus4": {"intervals": [0, 5, 7], "degrees": ["1", "4", "5"], "group": "Common"},
    "Major 7": {"intervals": [0, 4, 7, 11], "degrees": ["1", "3", "5", "7"], "group": "Common"},
    "Minor 7": {"intervals": [0, 3, 7, 10], "degrees": ["1", "b3", "5", "b7"], "group": "Common"},
    "Dominant 7": {"intervals": [0, 4, 7, 10], "degrees": ["1", "3", "5", "b7"], "group": "Blues"},
    "Dominant 9": {"intervals": [0, 4, 7, 10, 2], "degrees": ["1", "3", "5", "b7", "9"], "group": "Blues"},
    "Dominant 13": {"intervals": [0, 4, 7, 10, 2, 9], "degrees": ["1", "3", "5", "b7", "9", "13"], "group": "Blues"},
    "7#9": {"intervals": [0, 4, 7, 10, 3], "degrees": ["1", "3", "5", "b7", "#9"], "group": "Blues"},
    "m7b5": {"intervals": [0, 3, 6, 10], "degrees": ["1", "b3", "b5", "b7"], "group": "Common"},
    "Diminished 7": {"intervals": [0, 3, 6, 9], "degrees": ["1", "b3", "b5", "bb7"], "group": "Common"},
}

DEGREE_COLORS = {
    "1": "#e63946",
    "b3": "#457b9d",
    "3": "#2a9d8f",
    "5": "#f4a261",
    "b5": "#6d597a",
    "#5": "#ff7f51",
    "b7": "#6a994e",
    "7": "#577590",
    "2": "#8d99ae",
    "4": "#bc6c25",
    "9": "#8d99ae",
    "13": "#dda15e",
    "#9": "#264653",
    "bb7": "#5e548e",
}


def parse_root_label(root_label: str) -> str:
    if "/" in root_label:
        return root_label.split("/")[1]
    return root_label


def pc_to_note(pc: int, prefer_flats: bool = False) -> str:
    labels = NOTE_LABELS_FLAT if prefer_flats else NOTE_LABELS_SHARP
    return labels[pc % 12]


def spell_chord_tone(root_spelling: str, root_pc: int, semis: int, degree: str) -> str:
    root_letter = root_spelling[0]
    root_idx = LETTERS.index(root_letter)
    step = DEGREE_TO_STEP[degree]
    target_letter = LETTERS[(root_idx + step) % 7]

    desired_pc = (root_pc + semis) % 12
    target_natural_pc = NATURAL_PCS[target_letter]
    delta = (desired_pc - target_natural_pc + 12) % 12
    if delta > 6:
        delta -= 12

    accidental = OFFSET_TO_ACCIDENTAL.get(delta)
    if accidental is None:
        return pc_to_note(desired_pc, prefer_flats=("b" in root_spelling))
    return f"{target_letter}{accidental}"


def build_chord(root_pc: int, chord_name: str, root_spelling: str) -> List[Dict[str, object]]:
    data = CHORDS[chord_name]
    intervals = data["intervals"]
    degrees = data["degrees"]
    tones = []
    for semis, degree in zip(intervals, degrees):
        tones.append(
            {
                "note": spell_chord_tone(root_spelling, root_pc, semis, degree),
                "degree": degree,
                "pc": (root_pc + semis) % 12,
            }
        )
    return tones


def build_fretboard_df(root_pc: int, chord_name: str, root_spelling: str, max_fret: int) -> pd.DataFrame:
    tones = build_chord(root_pc, chord_name, root_spelling)
    tone_map = {tone["pc"]: (tone["note"], tone["degree"]) for tone in tones}

    rows = []
    for string_idx, open_pc in enumerate(STRING_OPEN_PCS):
        for fret in range(max_fret + 1):
            fret_pc = (open_pc + fret) % 12
            if fret_pc in tone_map:
                note, degree = tone_map[fret_pc]
                rows.append(
                    {
                        "string": string_idx + 1,
                        "string_label": STRING_LABELS[string_idx],
                        "fret": fret,
                        "note": note,
                        "degree": degree,
                        "is_root": degree == "1",
                    }
                )
    return pd.DataFrame(rows)


def fretboard_chart(df: pd.DataFrame, max_fret: int) -> go.Figure:
    fig = go.Figure()
    for degree in df["degree"].unique():
        subset = df[df["degree"] == degree]
        fig.add_trace(
            go.Scatter(
                x=subset["fret"],
                y=subset["string"],
                mode="markers+text",
                text=subset["note"],
                textposition="middle center",
                name=degree,
                marker=dict(
                    size=[24 if root else 18 for root in subset["is_root"]],
                    color=DEGREE_COLORS.get(degree, "#495057"),
                    line=dict(color="white", width=1),
                    symbol="circle",
                ),
                hovertemplate="String %{y}, Fret %{x}<br>%{text} (%{fullData.name})<extra></extra>",
            )
        )

    fig.update_layout(
        title="Chord tones on the fretboard",
        xaxis=dict(title="Fret", tickmode="linear", dtick=1, range=[-0.5, max_fret + 0.5]),
        yaxis=dict(
            title="String",
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5, 6],
            ticktext=STRING_LABELS,
        ),
        legend_title="Degree",
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
    )
    return fig


def random_quiz_question(pool: List[str]) -> Dict[str, object]:
    chord_name = random.choice(pool)
    root_label, root_pc = random.choice(ROOT_OPTIONS)
    root_spelling = parse_root_label(root_label)

    tones = build_chord(root_pc, chord_name, root_spelling)
    target = random.choice(tones)
    target_note = target["note"]
    target_degree = target["degree"]

    distractors = [n for n in QUIZ_NOTE_OPTIONS if n != target_note]
    options = random.sample(distractors, 3) + [target_note]
    random.shuffle(options)

    return {
        "chord_name": chord_name,
        "root_label": root_spelling,
        "root_pc": root_pc,
        "target_degree": target_degree,
        "correct_note": target_note,
        "options": options,
    }


st.title("Guitar Chord Tone Trainer")
st.caption(
    "Learn where chord tones live on the fretboard, with emphasis on common chord types and blues harmony."
)

with st.sidebar:
    st.header("Practice setup")
    root_label = st.selectbox("Root note", [label for label, _ in ROOT_OPTIONS], index=0)
    root_pc = dict(ROOT_OPTIONS)[root_label]
    root_spelling = parse_root_label(root_label)

    chord_names = list(CHORDS.keys())
    default_chord = chord_names.index("Dominant 7") if "Dominant 7" in chord_names else 0
    chord_name = st.selectbox("Chord quality", chord_names, index=default_chord)
    max_fret = st.slider("Show frets 0 to", min_value=5, max_value=18, value=12)

chord_tones = build_chord(root_pc, chord_name, root_spelling)
chord_formula = " - ".join([f"{tone['degree']} ({tone['note']})" for tone in chord_tones])

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"{root_spelling} {chord_name}")
    st.write(f"Chord formula: **{chord_formula}**")

    fret_df = build_fretboard_df(root_pc, chord_name, root_spelling, max_fret)
    chart = fretboard_chart(fret_df, max_fret)
    st.plotly_chart(chart, use_container_width=True)

with col2:
    st.subheader("Tone Targets")
    tone_table = pd.DataFrame(
        [{"Note": tone["note"], "Degree": tone["degree"]} for tone in chord_tones]
    )
    st.dataframe(tone_table, hide_index=True, use_container_width=True)

    st.markdown("**How to practice**")
    st.write(
        "1. Start with only roots and 3rds.\n"
        "2. Add 5ths, then 7ths.\n"
        "3. Improvise while landing on the 3rd or b7 when chords change."
    )

st.markdown("---")
st.subheader("Blues I-IV-V helper")
st.markdown("**Quick notes**")
st.write(
    "- **I7, IV7, V7** are the 3 core chords in a 12-bar blues.\n"
    "- **Function** shows where each chord sits in the key (home, away, tension).\n"
    "- **Guide tones (3 and b7)** define the chord sound and connect smoothly between changes.\n"
    "- Practice idea: target only 3 and b7 on each change, then add root and 5th."
)
key_label = st.selectbox("12-bar blues key", [label for label, _ in ROOT_OPTIONS], index=7)
key_pc = dict(ROOT_OPTIONS)[key_label]
key_spelling = parse_root_label(key_label)
prefer_flats = "b" in key_spelling and "#" not in key_spelling

blues_roots = [("I7", key_pc), ("IV7", (key_pc + 5) % 12), ("V7", (key_pc + 7) % 12)]
blues_rows = []
for func, pc in blues_roots:
    chord_root = pc_to_note(pc, prefer_flats=prefer_flats)
    tones = build_chord(pc, "Dominant 7", chord_root)
    blues_rows.append(
        {
            "Chord": f"{chord_root}7",
            "Function": func,
            "Chord tones": ", ".join([f"{tone['note']} ({tone['degree']})" for tone in tones]),
            "Guide tones": ", ".join(
                [
                    f"{tone['note']} ({tone['degree']})"
                    for tone in tones
                    if tone["degree"] in {"3", "b7"}
                ]
            ),
        }
    )

st.dataframe(pd.DataFrame(blues_rows), hide_index=True, use_container_width=True)
st.caption("Guide tones (3 and b7) are the strongest notes for hearing blues chord movement.")

st.markdown("---")
st.subheader("Quiz mode: chord tone recognition")
quiz_scope = st.radio(
    "Question pool",
    ["Selected chord only", "All common chords", "Blues chords"],
    horizontal=True,
)

if quiz_scope == "Selected chord only":
    pool = [chord_name]
elif quiz_scope == "All common chords":
    pool = [name for name, info in CHORDS.items() if info["group"] == "Common"]
else:
    pool = [name for name, info in CHORDS.items() if info["group"] == "Blues"]

pool_signature = "|".join(sorted(pool))

if "quiz_pool_signature" not in st.session_state:
    st.session_state.quiz_pool_signature = pool_signature
if "quiz_question" not in st.session_state:
    st.session_state.quiz_question = random_quiz_question(pool)
if "quiz_correct" not in st.session_state:
    st.session_state.quiz_correct = 0
if "quiz_attempts" not in st.session_state:
    st.session_state.quiz_attempts = 0

if st.session_state.quiz_pool_signature != pool_signature:
    st.session_state.quiz_pool_signature = pool_signature
    st.session_state.quiz_question = random_quiz_question(pool)
    st.session_state.pop("quiz_answer", None)

if st.button("New question"):
    st.session_state.quiz_question = random_quiz_question(pool)
    st.session_state.pop("quiz_answer", None)

q = st.session_state.quiz_question
st.write(f"In **{q['root_label']} {q['chord_name']}**, which note is the **{q['target_degree']}**?")
answer = st.radio("Choose one", q["options"], key="quiz_answer")

if st.button("Check answer"):
    st.session_state.quiz_attempts += 1
    if answer == q["correct_note"]:
        st.session_state.quiz_correct += 1
        st.success("Correct.")
    else:
        st.error(f"Not quite. Correct answer: {q['correct_note']}")

attempts = st.session_state.quiz_attempts
correct = st.session_state.quiz_correct
accuracy = 0.0 if attempts == 0 else correct / attempts

c1, c2, c3 = st.columns(3)
c1.metric("Correct", correct)
c2.metric("Attempts", attempts)
c3.metric("Accuracy", f"{accuracy:.0%}")

st.caption("Tip: if you play blues, drill Dominant 7, Dominant 9, and 7#9 first in every key.")
