"""
quiz_agent.py
-------------
The core agent: combines ChromaDB retrieval + live web search context,
grounds an LLM prompt in that retrieved evidence, and asks the model to
produce a structured multiple-choice sports quiz as JSON.

Design notes:
  - The LLM is instructed to answer ONLY from the supplied context, and to
    explicitly say if it cannot verify a fact, which is the main anti-
    hallucination guardrail (grounding + refusal-to-fabricate instruction).
  - Every call includes a `seed_hint` (random topic angle) so regenerate
    requests produce genuinely different questions instead of near-duplicates.
  - Uses Groq's free API (llama-3.3-70b-versatile) -- genuinely free tier, no
    credit card or billing link required. Swap MODEL / the client call if you
    want to use a different provider -- everything downstream only depends on
    _parse_json() receiving the model's raw text output.
"""

import json
import os
import random
from typing import Dict, List

from groq import Groq

from .knowledge_base import get_knowledge_base
from .web_search import search_web, results_to_context

MODEL = "llama-3.3-70b-versatile"

SPORTS = ["Cricket", "Football", "Tennis", "Badminton", "Basketball"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

TOPIC_ANGLES = [
    "famous tournament winners",
    "record holders and statistics",
    "rules and scoring systems",
    "iconic matches and moments",
    "player achievements and milestones",
    "team and country history",
]

SYSTEM_PROMPT = """You are a meticulous sports quiz writer. You must ONLY use
facts present in the provided CONTEXT (retrieved knowledge + web search results)
to write quiz questions. Do not invent statistics, dates, scores, or names that
are not explicitly supported by the CONTEXT. If the CONTEXT is insufficient to
write a fully verifiable question on a topic, choose a different angle that IS
supported by the CONTEXT rather than guessing.

Always respond with valid JSON only -- no markdown fences, no commentary."""


class QuizAgent:
    def __init__(self):
        self.kb = get_knowledge_base()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file. "
                "Get a free key at https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)

    def _gather_context(self, sport: str, use_web_search: bool = True) -> str:
        angle = random.choice(TOPIC_ANGLES)

        # 1. Retrieve grounded facts from ChromaDB, biased toward a topic angle
        kb_facts = self.kb.retrieve(sport, query=angle, n_results=6)
        kb_context = "\n".join(f"- {fact}" for fact in kb_facts)

        # 2. Pull fresh info from the web to supplement/verify recent events
        web_context = ""
        if use_web_search:
            query = f"{sport} {angle} latest facts 2026"
            results = search_web(query, max_results=5)
            web_context = results_to_context(results)
            # Feed anything useful straight back into the vector DB so future
            # quizzes benefit from it too (keeps the knowledge base "fresh").
            if results:
                new_facts = [
                    {
                        "id": f"web_{sport.lower()}_{abs(hash(r['url']))}",
                        "sport": sport,
                        "text": r["snippet"],
                    }
                    for r in results
                    if r["snippet"]
                ]
                self.kb.add_facts(new_facts)

        context = "STORED KNOWLEDGE:\n" + (kb_context or "(none found)")
        if web_context:
            context += "\n\nRECENT WEB SEARCH RESULTS:\n" + web_context
        return context, angle

    def generate_quiz(
        self,
        sport: str,
        difficulty: str,
        num_questions: int = 5,
        use_web_search: bool = True,
    ) -> Dict:
        if sport not in SPORTS:
            raise ValueError(f"Unsupported sport: {sport}")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        context, angle = self._gather_context(sport, use_web_search=use_web_search)

        user_prompt = f"""CONTEXT:
{context}

TASK:
Write {num_questions} multiple-choice quiz questions about {sport}, at {difficulty}
difficulty, loosely themed around "{angle}" where the context supports it.

Rules:
- Each question must have exactly 4 options (A, B, C, D).
- Exactly one option is correct.
- Every question and its correct answer must be traceable to the CONTEXT above.
- Include a short (1-2 sentence) explanation for the correct answer.
- Vary question topics across the set -- don't repeat the same fact twice.
- Difficulty guide: Easy = widely known facts; Medium = needs closer sport
  knowledge; Hard = specific stats, dates, or lesser-known details.

Respond with JSON in exactly this shape:
{{
  "sport": "{sport}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "string",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "correct_answer": "A",
      "explanation": "string"
    }}
  ]
}}"""

        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content
        return self._parse_json(raw_text)

    @staticmethod
    def _parse_json(raw_text: str) -> Dict:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model did not return valid JSON: {e}\nRaw output:\n{raw_text}"
            )


if __name__ == "__main__":
    agent = QuizAgent()
    quiz = agent.generate_quiz("Badminton", "Medium", num_questions=3)
    print(json.dumps(quiz, indent=2))
