"""Modular Natural Language Generation (NLG) engine for therapist-style responses."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.config import load_config
from src.utils import get_logger

logger = get_logger("nlg_engine")


@dataclass
class NLGConfig:
    """Configuration for NLG engine."""

    style: str = "CBT-therapeutic"
    max_retries: int = 3
    temperature: float = 0.6
    use_safety_check: bool = True


def build_prompt(
    user_text: str,
    emotional_state: Dict[str, Any],
    nlg_cfg: Optional[NLGConfig] = None,
) -> str:
    """
    Build a structured CBT-style prompt for an LLM, combining:
    - user input
    - primary emotion
    - depression risk
    - safety flag
    - recommended tone

    Args:
        user_text: User's input text
        emotional_state: Dictionary with emotional state information
        nlg_cfg: Optional NLG configuration

    Returns:
        Formatted prompt string
    """
    if nlg_cfg is None:
        nlg_cfg = NLGConfig()

    # Extract emotional state information safely
    primary_emotion = emotional_state.get("primary_emotion", "unknown")
    primary_emotion_confidence = emotional_state.get("primary_emotion_confidence", 0.0)
    depression_risk = emotional_state.get("depression_risk")
    safety_flag = emotional_state.get("safety_flag", False)

    # Build prompt
    prompt_parts = [
        "You are a supportive, CBT-informed assistant helping someone process their emotions.",
        "Your role is to validate feelings, reflect concerns, and gently explore thoughts.",
        "You are NOT a doctor and should NOT make diagnoses.",
        "",
        "USER INPUT:",
        f'"{user_text}"',
        "",
        "EMOTIONAL STATE:",
        f"- Primary emotion: {primary_emotion} (confidence: {primary_emotion_confidence:.2%})",
    ]

    if depression_risk is not None:
        prompt_parts.append(f"- Depression risk: {depression_risk:.2%}")
    else:
        prompt_parts.append("- Depression risk: Not assessed")

    prompt_parts.append(f"- Safety flag: {safety_flag}")

    prompt_parts.extend(
        [
            "",
            "INSTRUCTIONS:",
            "1. Validate the user's feelings with empathy and understanding.",
            "2. Reflect and summarize their concerns to show you understand.",
            "3. Gently explore their thoughts using CBT principles (e.g., 'What evidence supports this thought?').",
            "4. Provide 1-2 small, actionable steps based on CBT techniques.",
            "5. Use a gentle, validating, CBT-informed tone throughout.",
        ]
    )

    if safety_flag:
        prompt_parts.extend(
            [
                "",
                "IMPORTANT - SAFETY:",
                "The user may be experiencing significant distress. Include a supportive safety message",
                "that encourages them to reach out to mental health professionals or crisis resources",
                "if they feel overwhelmed, lonely, or unsafe. Be warm and non-judgmental.",
            ]
        )

    if depression_risk is not None and depression_risk >= 0.7:
        prompt_parts.extend(
            [
                "",
                "CRISIS SUPPORT DISCLAIMER:",
                "Include a clear disclaimer that this is not a medical diagnosis and that",
                "professional mental health support is recommended.",
            ]
        )

    prompt_parts.extend(
        [
            "",
            "RESPONSE FORMAT:",
            "- Start with a warm, validating paragraph",
            "- Include a brief reflection of their concerns",
            "- Provide 2 small CBT-based actionable steps",
            "- End with a safety note if safety_flag is True",
            "",
            "Generate your response now:",
        ]
    )

    return "\n".join(prompt_parts)


def call_llm(prompt: str) -> str:
    """
    Placeholder for real LLM inference.

    For now, return a formatted pseudo-therapist response meant to simulate
    what a real LLM would generate.

    Args:
        prompt: Input prompt for the LLM

    Returns:
        Simulated therapist-style response
    """
    # Check if safety flag is mentioned in prompt
    has_safety_flag = "safety_flag: True" in prompt or "SAFETY:" in prompt
    has_crisis_disclaimer = "CRISIS SUPPORT DISCLAIMER:" in prompt

    # Extract primary emotion from prompt if possible
    primary_emotion = "these feelings"
    if "Primary emotion:" in prompt:
        try:
            emotion_line = [line for line in prompt.split("\n") if "Primary emotion:" in line][0]
            primary_emotion = emotion_line.split(":")[1].split("(")[0].strip().lower()
        except:
            pass

    # Build simulated response
    response_parts = [
        f"I hear that you're experiencing {primary_emotion} right now, and I want you to know that "
        "your feelings are completely valid. It takes courage to acknowledge and share what you're "
        "going through, and I appreciate you trusting me with this.",
        "",
        "From what you've shared, it sounds like you're dealing with some difficult thoughts and "
        "emotions. Sometimes when we're in the midst of challenging feelings, our minds can focus "
        "on the most difficult aspects of a situation. Let's gently explore what might be happening here.",
        "",
        "Here are a couple of small steps you might consider:",
        "",
        "1. **Thought exploration**: When you notice a difficult thought, try asking yourself: "
        "'What evidence supports this thought? What evidence challenges it? What would I tell a "
        "friend who had this same thought?' This can help create some space between you and the thought.",
        "",
        "2. **Small action**: Think of one very small, manageable action you could take today that "
        "aligns with your values or brings a moment of comfort. It doesn't need to be big - even "
        "something like taking a few deep breaths, stepping outside for a moment, or reaching out "
        "to someone you trust can make a difference.",
    ]

    if has_safety_flag or has_crisis_disclaimer:
        response_parts.extend(
            [
                "",
                "**Important note**: This conversation is not a substitute for professional mental "
                "health care. If you ever feel overwhelmed, lonely, or unsafe, please consider reaching "
                "out to a mental health professional or crisis resource near you. You don't have to "
                "face these challenges alone, and there are people who want to help.",
            ]
        )

    if has_crisis_disclaimer:
        response_parts.extend(
            [
                "",
                "Please remember that this is not a medical diagnosis. If you're experiencing "
                "significant distress, professional mental health support is strongly recommended.",
            ]
        )

    return "\n".join(response_parts)


def generate_nlg_response(
    user_text: str,
    emotional_state: Dict[str, Any],
    nlg_cfg: Optional[NLGConfig] = None,
) -> Dict[str, Any]:
    """
    High-level function to generate NLG response.

    Args:
        user_text: User's input text
        emotional_state: Dictionary with emotional state information
        nlg_cfg: Optional NLG configuration

    Returns:
        Dictionary with:
            - response: Generated response text
            - prompt_used: Prompt that was used
            - primary_emotion: Primary emotion detected
            - depression_risk: Depression risk score or None
            - safety_flag: Boolean safety flag
    """
    if nlg_cfg is None:
        nlg_cfg = NLGConfig()

    try:
        # Build prompt
        prompt = build_prompt(user_text, emotional_state, nlg_cfg)

        # Call LLM (placeholder for now)
        final_response = call_llm(prompt)

        # Extract state information safely
        primary_emotion = emotional_state.get("primary_emotion")
        depression_risk = emotional_state.get("depression_risk")
        safety_flag = emotional_state.get("safety_flag", False)

        return {
            "response": final_response,
            "prompt_used": prompt,
            "primary_emotion": primary_emotion,
            "depression_risk": depression_risk,
            "safety_flag": safety_flag,
        }

    except Exception as e:
        logger.error(f"Failed to generate NLG response: {e}")
        # Return a fallback response
        return {
            "response": (
                "I'm sorry, I'm having trouble processing your message right now. "
                "Please try again, or consider reaching out to a mental health professional "
                "if you need immediate support."
            ),
            "prompt_used": "",
            "primary_emotion": emotional_state.get("primary_emotion"),
            "depression_risk": emotional_state.get("depression_risk"),
            "safety_flag": emotional_state.get("safety_flag", False),
        }


def build_emotional_state(
    emotion_label: Optional[str] = None,
    emotion_confidence: Optional[float] = None,
    depression_prob: Optional[float] = None,
) -> str:
    """
    Build a human-readable description of the user's emotional state from available modalities.
    
    Prioritization:
    1. Text emotion (highest priority) - if emotion_label exists, use it as primary
    2. Audio depression probability - incorporate meaningfully alongside or instead of text
    3. Video-only case - return soft, uncertain affect summary if only video available
    
    Args:
        emotion_label: Primary emotion detected from text analysis (e.g., "sad", "anxious")
        emotion_confidence: Confidence score for the text emotion (0.0 to 1.0)
        depression_prob: Depression probability from audio analysis (0.0 to 1.0)
    
    Returns:
        One clean sentence combining available modalities. Never returns "no clear emotional signal".
    """
    # Priority 1: Text emotion exists - use as primary
    if emotion_label:
        # Normalize emotion label (lowercase, strip whitespace)
        emotion_clean = str(emotion_label).strip().lower()
        
        # Build base sentence with emotion
        if emotion_confidence is not None and emotion_confidence > 0:
            conf_str = "high" if emotion_confidence >= 0.7 else "moderate" if emotion_confidence >= 0.4 else "some"
            base_sentence = f"Primary emotion detected is {emotion_clean} ({conf_str} confidence)"
        else:
            base_sentence = f"Primary emotion detected is {emotion_clean}"
        
        # Incorporate depression probability if available
        if depression_prob is not None:
            try:
                if depression_prob >= 0.7:
                    dep_severity = "elevated"
                elif depression_prob >= 0.4:
                    dep_severity = "moderate"
                else:
                    dep_severity = "mild"
                return f"{base_sentence}, with {dep_severity} depression risk indicated by audio analysis."
            except Exception:
                return f"{base_sentence}."
        
        return f"{base_sentence}."
    
    # Priority 2: Audio depression probability exists (no text emotion)
    if depression_prob is not None:
        try:
            if depression_prob >= 0.7:
                severity = "high"
                descriptor = "significant"
            elif depression_prob >= 0.4:
                severity = "moderate"
                descriptor = "notable"
            else:
                severity = "mild"
                descriptor = "some"
            
            return (
                f"Audio analysis indicates {severity} depression risk ({descriptor} probability), "
                f"suggesting possible emotional distress that may benefit from supportive attention."
            )
        except Exception:
            return (
                "Audio analysis suggests some level of emotional distress that may benefit from "
                "supportive attention."
            )
    
    # Priority 3: Only video embedding or no clear signals (soft, uncertain summary)
    # When neither text emotion nor audio depression probability are available,
    # return a gentle, uncertain affect summary appropriate for video-only scenarios
    return (
        "Visual cues suggest possible emotional expression, though interpretations from facial "
        "features alone are inherently uncertain and may not fully reflect internal experience."
    )


__all__ = [
    "NLGConfig",
    "build_prompt",
    "call_llm",
    "generate_nlg_response",
    "build_emotional_state",
]

