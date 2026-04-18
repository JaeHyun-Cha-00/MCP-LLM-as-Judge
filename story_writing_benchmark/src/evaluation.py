import json
import re
from dataclasses import dataclass
from clients import WolverineClient

# Categories matching q1-q15 from lars1234/story_writing_benchmark
# Reference scores use a 1-5 scale; judge scores use 0-20 scale
EVALUATION_CATEGORIES = [
    "Grammar, Spelling, and Punctuation Quality",       # q1
    "Clarity and Understandability",                    # q2
    "Logical Connection Between Events and Ideas",      # q3
    "Scene Construction and Purpose",                   # q4
    "Internal Consistency",                             # q5
    "Character Consistency",                            # q6
    "Character Motivation and Actions",                 # q7
    "Sentence Pattern Variety",                         # q8
    "Avoidance of Clichés and Overused Phrases",        # q9
    "Natural Dialogue",                                 # q10
    "Avoidance of Predictable Narrative Tropes",        # q11
    "Character Depth and Dimensionality",               # q12
    "Realistic Character Interactions",                 # q13
    "Ability to Hold Reader Interest",                  # q14
    "Satisfying Plot Resolution",                       # q15
]

EVALUATION_SYSTEM_PROMPT = (
    "You are a literary critic. Always respond with JSON containing the key "
    '"score" (a number from 0.0 to 20.0, can include one decimal place like 15.5).'
)


@dataclass
class EvaluationResult:
    category: str
    score: float

    def to_dict(self) -> dict:
        return {"category": self.category, "score": self.score}


class StoryEvaluator:
    """Evaluate stories across literary quality categories."""

    def __init__(self, client: WolverineClient):
        self._client = client

    def evaluate_all_categories(self, story: str) -> dict[str, EvaluationResult]:
        """Evaluate a story across all categories in a single LLM call."""
        category_list = "\n".join([f"  - {cat}" for cat in EVALUATION_CATEGORIES])

        combined_prompt = (
            f"Evaluate the following story across all these categories. "
            f"For each category, provide a score from 0.0 to 20.0 (can include one decimal place like 15.5). "
            f"Higher scores indicate better quality.\n\n"
            f"Categories:\n{category_list}\n\n"
            f"Respond with JSON containing a 'scores' object where each key is the category name and the value is the score.\n"
            f"Example format: {{\"scores\": {{\"Grammar, Spelling, and Punctuation Quality\": 16.5, \"Natural Dialogue\": 14.0, ...}}}}\n\n"
            f"Story:\n{story}"
        )

        response = self._client.chat(
            system_prompt="You are a literary critic. Always respond with valid JSON containing a 'scores' object with category names as keys and numeric scores (0.0-20.0) as values.",
            user_prompt=combined_prompt,
        )

        results = {}
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            payload = json.loads(cleaned)
            scores = payload.get("scores", {})

            for category in EVALUATION_CATEGORIES:
                score = scores.get(category)
                if score is None:
                    for key, value in scores.items():
                        if category.lower() == key.lower() or category.lower() in key.lower() or key.lower() in category.lower():
                            score = value
                            break

                if score is None:
                    score = 0.0
                else:
                    try:
                        score = max(0.0, min(20.0, float(score)))
                    except (ValueError, TypeError):
                        score = 0.0

                results[category] = EvaluationResult(category=category, score=score)

        except json.JSONDecodeError:
            print("[WARNING] Failed to parse combined evaluation, falling back to individual calls")
            for category in EVALUATION_CATEGORIES:
                user_prompt = f"Evaluate the following story focusing strictly on the category: {category}.\n\nStory:\n{story}"
                resp = self._client.chat(
                    system_prompt=EVALUATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                score = _parse_single_score(resp)
                results[category] = EvaluationResult(
                    category=category,
                    score=max(0.0, min(20.0, score if score is not None else 0.0)),
                )

        return results


def _parse_single_score(response: str) -> float | None:
    response = response.strip()
    if not response:
        return None
    try:
        payload = json.loads(response)
        return float(payload["score"]) if "score" in payload else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"(?<!\d)(20(\.\d)?|1[0-9](\.\d)?|[0-9](\.\d)?)(?!\d)", response)
    return float(match.group(0)) if match else None
