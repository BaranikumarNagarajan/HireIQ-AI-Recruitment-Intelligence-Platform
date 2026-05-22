"""RAG Evaluation Pipeline — sequential, single LLM via get_llm_response()."""
import logging
from typing import Any, Dict

import pandas as pd

from config import GROQ_API_KEY
from rag.retriever import retrieve_legal_context
from utils.llm_client import parse_llm_json_response

_EVAL_MODEL = "llama-3.1-8b-instant"  # high token/min limit — avoids 429 on free tier


def get_llm_response(prompt: str, temperature: float = 0.2) -> str:
    """Eval-specific LLM call using the fast 8b model."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        chat = client.chat.completions.create(
            model=_EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        return chat.choices[0].message.content or ""
    except Exception as exc:
        return f"Error: {exc}"

logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    {
        "question": "What are the key GDPR requirements for processing candidate CV data?",
        "ground_truth": "Under GDPR, organizations must have a lawful basis for processing personal data, provide privacy notices to candidates, ensure data minimization, implement security measures, and respect data subject rights including access, rectification, and erasure.",
    },
    {
        "question": "How long can employers retain candidate data under data protection laws?",
        "ground_truth": "Employers should retain candidate data only for as long as necessary for the recruitment process. Typically 6 months to 1 year after the position is filled, unless the candidate consents to longer retention or there is a legal requirement.",
    },
    {
        "question": "What constitutes discrimination in hiring practices?",
        "ground_truth": "Discrimination includes decisions based on protected characteristics such as race, gender, age, disability, religion, or sexual orientation, including indirect discrimination through seemingly neutral criteria.",
    },
    {
        "question": "What disclosures must employers make to candidates during recruitment?",
        "ground_truth": "Employers must disclose how candidate data will be used, stored, and retained; the legal basis for processing; and candidate rights under data protection laws.",
    },
    {
        "question": "What are the requirements for automated decision-making in recruitment?",
        "ground_truth": "GDPR Article 22 requires candidates to be informed of automated decision-making and profiling. Candidates have the right to human intervention and to contest decisions.",
    },
]

_SCORE_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Score the following on three metrics, each from 0.0 to 1.0:
- faithfulness: Is the answer factually grounded in the context? (1.0 = fully grounded, 0.0 = contradicts context)
- answer_relevancy: Does the answer actually address the question? (1.0 = fully relevant, 0.0 = off-topic)
- context_precision: Is the retrieved context relevant to the question? (1.0 = highly relevant, 0.0 = irrelevant)

Question: {question}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}

Reply with ONLY valid JSON, no explanation:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}}"""


def _score_question(question: str, context: str, answer: str, ground_truth: str) -> Dict[str, float]:
    """Score one Q&A pair with a single LLM call. Returns metric dict."""
    prompt = _SCORE_PROMPT.format(
        question=question, context=context[:1500], answer=answer[:800], ground_truth=ground_truth
    )
    try:
        raw = get_llm_response(prompt, temperature=0.0)
        scores = parse_llm_json_response(raw)
        return {
            "faithfulness": float(scores.get("faithfulness", 0.5)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0.5)),
            "context_precision": float(scores.get("context_precision", 0.5)),
        }
    except Exception as exc:
        logger.warning("Scoring parse failed (%s) — using neutral 0.5", exc)
        return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}


def run_evaluation() -> Dict[str, Any]:
    """Run sequential RAG evaluation — 2 LLM calls per question, no concurrency."""
    try:
        records = []

        for test_case in TEST_QUESTIONS:
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            # Step 1 — retrieve context
            retrieved = retrieve_legal_context(question, n_results=3)
            context_texts = [c["text"] for c in retrieved]
            context_str = "\n\n".join(context_texts)

            # Step 2 — generate answer
            answer_prompt = (
                f"Based on the following legal context, answer this recruitment compliance question concisely.\n\n"
                f"Question: {question}\n\nContext:\n{context_str}"
            )
            answer = get_llm_response(answer_prompt)

            # Step 3 — score with a single LLM call
            scores = _score_question(question, context_str, answer, ground_truth)

            records.append({
                "question": question,
                "answer": answer,
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "context_precision": scores["context_precision"],
            })
            logger.info("Evaluated: %s… → %s", question[:50], scores)

        df = pd.DataFrame(records)
        return {
            "per_question_scores": df.to_dict("records"),
            "average_faithfulness": float(df["faithfulness"].mean()),
            "average_answer_relevancy": float(df["answer_relevancy"].mean()),
            "average_context_precision": float(df["context_precision"].mean()),
            "model_used": _EVAL_MODEL,
        }

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _score_label(score: float) -> tuple:
    """Return (emoji, colour, label) based on score threshold."""
    if score >= 0.8:
        return "🟢", "green", "Excellent"
    if score >= 0.6:
        return "🟡", "orange", "Good"
    if score >= 0.4:
        return "🟠", "darkorange", "Fair"
    return "🔴", "red", "Poor"


def display_eval_results(results: Dict[str, Any]) -> None:
    """Display RAG evaluation results with clear visuals."""
    import streamlit as st

    if "error" in results:
        st.error(f"Evaluation failed: {results['error']}")
        return

    model = results.get("model_used", "unknown")
    st.caption(f"Evaluated with: `{model}` via Groq")

    faith = results["average_faithfulness"]
    relevancy = results["average_answer_relevancy"]
    precision = results["average_context_precision"]
    overall = (faith + relevancy + precision) / 3

    # ── Overall score banner ──────────────────────────────────────────────────
    emoji, colour, label = _score_label(overall)
    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:20px 28px;margin-bottom:16px;
                    border-left:6px solid {colour};">
          <span style="font-size:2rem;">{emoji}</span>
          <span style="font-size:1.6rem;font-weight:700;color:{colour};margin-left:12px;">
            Overall RAG Score: {overall:.2f} / 1.00 — {label}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Three metric cards ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    def metric_card(col, title, score, description):
        card_emoji, card_color, card_label = _score_label(score)
        col.markdown(
            f"""
            <div style="background:#12121f;border-radius:10px;padding:16px;
                        border:1px solid {card_color};text-align:center;">
              <div style="font-size:1.8rem;">{card_emoji}</div>
              <div style="font-size:1.1rem;font-weight:600;color:#eee;margin:4px 0;">{title}</div>
              <div style="font-size:2rem;font-weight:700;color:{card_color};">{score:.3f}</div>
              <div style="font-size:0.75rem;color:#aaa;margin-top:4px;">{card_label} · {description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_card(col1, "Faithfulness", faith, "Answer grounded in context")
    metric_card(col2, "Answer Relevancy", relevancy, "Answer addresses the question")
    metric_card(col3, "Context Precision", precision, "Retrieved context quality")

    st.divider()

    # ── Score interpretation guide ────────────────────────────────────────────
    with st.expander("📖 Score Interpretation Guide"):
        st.markdown("""
| Score | Rating | Meaning |
|-------|--------|---------|
| 0.80 – 1.00 | 🟢 Excellent | RAG system performing very well |
| 0.60 – 0.79 | 🟡 Good | Acceptable, minor gaps |
| 0.40 – 0.59 | 🟠 Fair | Needs improvement |
| 0.00 – 0.39 | 🔴 Poor | Significant issues with retrieval or generation |

- **Faithfulness**: is the answer factually grounded in the retrieved context?
- **Answer Relevancy**: does the answer address the question asked?
- **Context Precision**: are the retrieved chunks relevant to the question?
        """)

    # ── Per-question bar chart ────────────────────────────────────────────────
    st.subheader("Per-Question Score Comparison")

    df = pd.DataFrame(results["per_question_scores"])
    score_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    df["Q"] = [f"Q{i + 1}" for i in range(len(df))]
    chart_df = df.set_index("Q")[score_cols]
    st.bar_chart(chart_df, height=320)

    # ── Detailed per-question expandable rows ────────────────────────────────
    st.subheader("Detailed Scores per Question")
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        avg = (row["faithfulness"] + row["answer_relevancy"] + row["context_precision"]) / 3
        row_emoji, _, _ = _score_label(avg)
        with st.expander(f"{row_emoji} Q{idx}: {str(row['question'])[:90]}…"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Faithfulness", f"{row['faithfulness']:.3f}")
            mc2.metric("Answer Relevancy", f"{row['answer_relevancy']:.3f}")
            mc3.metric("Context Precision", f"{row['context_precision']:.3f}")
            if "answer" in row:
                answer_text = str(row['answer'])
                display = answer_text[:500] + "…" if len(answer_text) > 500 else answer_text
                st.markdown(f"**Generated Answer:** {display}")
