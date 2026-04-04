import json
import re
from dataclasses import dataclass
from clients import WolverineClient

EVALUATION_SYSTEM_PROMPT = (
    "You are a literary critic. Always respond with JSON containing the key "
    '"score" (a number from 0.0 to 20.0, can include one decimal place like 15.5).'
)

# Positive metrics: Higher is Better (0-20 scale)
POSITIVE_CATEGORIES = [
    "Adherence to Instructions",
    "Believable Character Actions",
    "Nuanced Characters",
    "Consistent Voice / Tone of Writing",
    "Imagery and Descriptive Quality",
    "Elegant Prose",
    "Emotionally Engaging",
    "Emotionally Complex",
    "Coherent",
    "Well-earned Lightness or Darkness",
    "Sentences Flow Naturally",
    "Overall Reader Engagement",
    "Overall Impression",
]

# Negative / Penalty metrics: Lower is Better (0-20 scale, lower scores indicate less of the problem)
NEGATIVE_CATEGORIES = [
    "Meandering",
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Overwrought",
    "Incongruent Ending Positivity",
    "Unearned Transformations",
]

STORY_EVALUATION_CATEGORIES = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES

CATEGORY_TYPES = {cat: "positive" for cat in POSITIVE_CATEGORIES}
CATEGORY_TYPES.update({cat: "negative" for cat in NEGATIVE_CATEGORIES})

def is_positive_category(category: str) -> bool:
    """Check if a category is a positive metric (higher is better)."""
    return CATEGORY_TYPES.get(category) == "positive"

def is_negative_category(category: str) -> bool:
    """Check if a category is a negative/penalty metric (lower is better)."""
    return CATEGORY_TYPES.get(category) == "negative"

def build_user_prompt(story: str, category: str) -> str:
    """Build the user prompt sent to the model."""
    return f"Evaluate the following story focusing strictly on the category: {category}.\n\nStory:\n{story}"


######## Data Classes ########
@dataclass
class EvaluationResult:
    """Evaluation Response for a single category."""

    category: str
    score: float

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "score": self.score,
        }
    
##### MCP-LLM Judge (Main) ########
class StoryEvaluator:
    """Evaluate stories for multiple literary categories."""

    def __init__(self, client: WolverineClient):
        """Initialize the evaluator with a Wolverine client."""
        self._client = client

    def evaluate_all_categories(self, story: str) -> dict[str, EvaluationResult]:
        """Evaluate a story across all categories in a single LLM call, then evaluate creativity. Much faster than individual calls."""
        # Build category list with type indicators
        positive_list = "\n".join([f"  - {cat} (POSITIVE: higher is better)" for cat in POSITIVE_CATEGORIES])
        negative_list = "\n".join([f"  - {cat} (NEGATIVE/PENALTY: lower is better)" for cat in NEGATIVE_CATEGORIES])
        
        combined_prompt = (
            f"Evaluate the following story across all these categories. "
            f"For each category, provide a score from 0.0 to 20.0 (can include one decimal place like 15.5).\n\n"
            f"POSITIVE METRICS (Higher scores are better):\n{positive_list}\n\n"
            f"NEGATIVE/PENALTY METRICS (Lower scores are better - score how much this problem exists):\n{negative_list}\n\n"
            f"For positive metrics: higher scores indicate better quality.\n"
            f"For negative metrics: lower scores indicate less of the problem (i.e., better quality).\n\n"
            f"Respond with JSON containing a 'scores' object where each key is the category name and the value is the score (number 0.0-20.0).\n"
            f"Example format: {{\"scores\": {{\"Adherence to Instructions\": 16.5, \"Meandering\": 4.0, ...}}}}\n\n"
            f"Story:\n{story}"
        )
        
        # Single LLM call for all categories
        response = self._client.chat(
            system_prompt="You are a literary critic. Always respond with valid JSON containing a 'scores' object with category names as keys and numeric scores (0.0-20.0) as values. Remember: positive metrics should have higher scores, negative/penalty metrics should have lower scores.",
            user_prompt=combined_prompt,
        )
        
        # Parse the response
        results = {}
        try:
            # Remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove closing ```
            cleaned_response = cleaned_response.strip()
            
            payload = json.loads(cleaned_response)
            scores = payload.get("scores", {})
            
            # Create EvaluationResult for each category
            for category in STORY_EVALUATION_CATEGORIES:
                score = scores.get(category)
                if score is None:
                    # Try to find by partial match (remove suffixes like " (POSITIVE)" from keys)
                    for key, value in scores.items():
                        # Remove common suffixes from response keys
                        cleaned_key = key.replace(" (POSITIVE)", "").replace(" (NEGATIVE/PENALTY)", "").strip()
                        if category.lower() == cleaned_key.lower() or category.lower() in cleaned_key.lower() or cleaned_key.lower() in category.lower():
                            score = value
                            break
                
                if score is None:
                    score = 0.0
                else:
                    try:
                        score = float(score)
                        # Clamp score to 0-20 range
                        if score < 0:
                            score = 0.0
                        elif score > 20:
                            score = 20.0
                    except (ValueError, TypeError):
                        score = 0.0
                
                results[category] = EvaluationResult(
                    category=category,
                    score=score,
                )
        except json.JSONDecodeError:
            # Fallback: if JSON parsing fails, try to extract scores individually
            print(f"[WARNING] Failed to parse combined evaluation, falling back to individual calls")
            for category in STORY_EVALUATION_CATEGORIES:
                user_prompt = build_user_prompt(story, category)
                response = self._client.chat(
                    system_prompt=EVALUATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                score = parse_response(response)
                if score is None:
                    score = 0.0
                else:
                    # Clamp score to 0-20 range
                    if score < 0:
                        score = 0.0
                    elif score > 20:
                        score = 20.0
                results[category] = EvaluationResult(
                    category=category,
                    score=score,
                )
        
        # Evaluate creativity based on all category results
        category_summary = "\n".join([
            f"- {cat}: {res.score}/20"
            for cat, res in results.items()
        ])
        creativity_prompt = (
            f"Based on the following evaluation scores across all categories, "
            f"what creativity score (0.0-20.0, can include one decimal place) would you give this story? "
            f"Consider how the story demonstrates originality, innovation, unique perspectives, and imaginative elements.\n\n"
            f"Evaluation Scores:\n{category_summary}\n\n"
            f"Original Story:\n{story}\n\n"
            f"Respond with JSON: {{\"score\": <number>}}"
        )
        creativity_response = self._client.chat(
            system_prompt="You are a literary critic. Always respond with valid JSON containing 'score' (number 0.0-20.0).",
            user_prompt=creativity_prompt,
        )
        creativity_score = parse_response(creativity_response)
        if creativity_score is None:
            creativity_score = 0.0
        else:
            # Clamp score to 0-20 range
            if creativity_score < 0:
                creativity_score = 0.0
            elif creativity_score > 20:
                creativity_score = 20.0
        results["Creativity"] = EvaluationResult(
            category="Creativity",
            score=creativity_score,
        )
        
        return results

    def evaluate_creativity(self, story: str) -> EvaluationResult:
        """Evaluate a story's creativity directly without breaking it into categories."""
        user_prompt = f"Evaluate the creativity of the following story. Consider originality, innovation, unique perspectives, and imaginative elements.\n\nStory:\n{story}"
        response = self._client.chat(
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        score = parse_response(response)
        if score is None:
            score = 0.0  # Default score if parsing fails
        else:
            # Clamp score to 0-20 range
            if score < 0:
                score = 0.0
            elif score > 20:
                score = 20.0
        return EvaluationResult(
            category="Creativity",
            score=score,
        )

    def analyze_creativity_difference(
        self, 
        story: str,
        standalone_creativity: EvaluationResult,
        all_categories_results: dict[str, EvaluationResult]
    ) -> dict:
        """Analyze which categories influenced the difference in creativity scores."""
        standalone_score = standalone_creativity.score
        contextual_score = all_categories_results.get("Creativity", standalone_creativity).score
        
        # If scores are the same (within 0.1 tolerance), return early
        if abs(standalone_score - contextual_score) < 0.1:
            return {
                "standalone_creativity_score": standalone_score,
                "contextual_creativity_score": contextual_score,
                "difference": 0.0,
                "influential_categories": []
            }
        
        # Build category summary (excluding Creativity itself)
        category_summary = "\n".join([
            f"- {cat}: {res.score}/20"
            for cat, res in all_categories_results.items()
            if cat != "Creativity"
        ])
        
        # List all available categories for reference with type indicators
        positive_ref = "\n".join([f"- {cat} (POSITIVE)" for cat in POSITIVE_CATEGORIES])
        negative_ref = "\n".join([f"- {cat} (NEGATIVE)" for cat in NEGATIVE_CATEGORIES])
        available_categories = f"Positive Metrics:\n{positive_ref}\n\nNegative Metrics:\n{negative_ref}"
        
        analysis_prompt = (
            f"Two different creativity scores were given for the same story:\n"
            f"- Standalone creativity score (evaluated without category context): {standalone_score}/20\n"
            f"- Contextual creativity score (evaluated after seeing all category results): {contextual_score}/20\n"
            f"- Difference: {abs(standalone_score - contextual_score):.1f} points\n\n"
            f"All category evaluation results:\n{category_summary}\n\n"
            f"Original Story:\n{story}\n\n"
            f"Available categories:\n{available_categories}\n\n"
            f"Please identify which specific categories influenced the change in creativity score. "
            f"You MUST only select from the categories listed above. Do not create new category names. "
            f"Respond with JSON containing: "
            f'"influential_categories" (list of category names from the available categories that most influenced the difference).'
        )
        
        analysis_response = self._client.chat(
            system_prompt="You are a literary analysis expert. Always respond with valid JSON.",
            user_prompt=analysis_prompt,
        )
        
        # Parse the analysis response
        try:
            analysis_data = json.loads(analysis_response)
            influential_categories = analysis_data.get("influential_categories", [])
            
            # Filter to only include valid categories from STORY_EVALUATION_CATEGORIES
            valid_categories = [
                cat for cat in influential_categories 
                if cat in STORY_EVALUATION_CATEGORIES
            ]
            
            return {
                "standalone_creativity_score": standalone_score,
                "contextual_creativity_score": contextual_score,
                "difference": round(abs(standalone_score - contextual_score), 1),
                "influential_categories": valid_categories
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "standalone_creativity_score": standalone_score,
                "contextual_creativity_score": contextual_score,
                "difference": round(abs(standalone_score - contextual_score), 1),
                "influential_categories": []
            }

###### Response Parsing ########
def parse_response(response: str) -> float | None:
    """Parse the model's response into a numeric score."""
    response = response.strip()
    if not response:
        return None

    # Try structured JSON response
    try:
        payload = json.loads(response)
        score = float(payload.get("score")) if "score" in payload else None
        return score
    except json.JSONDecodeError:
        pass

    # Fallback: extract numeric score from text
    # Pattern matches: 20.0-20.9, 20, 10.0-19.9, 10-19, 0.0-9.9, 0-9
    match = re.search(r"(?<!\d)(20(\.\d)?|1[0-9](\.\d)?|[0-9](\.\d)?)(?!\d)", response)
    if match:
        score_str = match.group(0)
        score = float(score_str)
        # Validate score is within 0-20 range
        if score < 0 or score > 20:
            return None
        return score
    
    return None