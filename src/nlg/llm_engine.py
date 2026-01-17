"""Purely local, multi-turn CBT therapist engine with no external API dependencies."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class PsychologistLLMConfig:
    """Configuration for the local CBT therapist engine."""
    
    max_history_turns: int = 6  # user+assistant messages = up to 6 pairs


def _check_crisis_language(user_message: str) -> bool:
    """
    Check if user message contains strong crisis language.
    
    Args:
        user_message: User's input message
    
    Returns:
        True if crisis language detected, False otherwise
    """
    crisis_keywords = [
        "kill myself",
        "suicide",
        "end it all",
        "end my life",
        "don't want to live",
        "not worth living",
        "better off dead",
        "taking my life",
    ]
    
    user_lower = user_message.lower()
    for keyword in crisis_keywords:
        if keyword in user_lower:
            return True
    return False


def _build_crisis_disclaimer() -> str:
    """
    Build a gentle crisis disclaimer message.
    
    Returns:
        Crisis support message string
    """
    return (
        "**Important: If you're having thoughts of harming yourself, please reach out for help immediately. "
        "You can contact a crisis hotline, local emergency services, or a mental health professional. "
        "You don't have to go through this alone, and there are people who want to help.**\n\n"
    )


def _clip_history(history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    """
    Clip history to the last max_turns pairs of user+assistant messages.
    
    Args:
        history: List of message dicts with 'role' and 'content' keys
        max_turns: Maximum number of user-assistant turn pairs to keep
    
    Returns:
        Clipped history list
    """
    if not history:
        return []
    
    # Keep the last max_turns * 2 messages (max_turns pairs)
    if len(history) <= max_turns * 2:
        return history
    
    return history[-(max_turns * 2):]


def _build_emotional_context(
    emotional_state: Optional[str],
    cbt_suggestions: Optional[List[str]],
    fused_vector: Optional[List[float]],
) -> str:
    """
    Build a short emotional context string from available information.
    
    Args:
        emotional_state: Optional emotional state description
        cbt_suggestions: Optional list of CBT suggestion strings
        fused_vector: Optional fused multimodal vector
    
    Returns:
        Context string summarizing emotional state and suggestions
    """
    parts = []
    
    if emotional_state:
        parts.append(f"Detected emotional state: {emotional_state}")
    
    if cbt_suggestions:
        suggestions_text = "; ".join(cbt_suggestions)
        parts.append(f"Relevant CBT suggestions: {suggestions_text}")
    
    if fused_vector is not None:
        parts.append(
            "Additional multimodal signals were detected (audio/video/text features used internally)."
        )
    
    return " | ".join(parts) if parts else "No specific emotional context available."


def _summarize_recent_history(history: List[Dict[str, str]]) -> str:
    """
    Create a brief summary of the last few conversation turns.
    
    Args:
        history: List of previous messages
    
    Returns:
        Short summary string (1-2 lines)
    """
    if not history:
        return "This is the start of the conversation."
    
    # Get the last user message and last assistant reply
    last_user = None
    last_assistant = None
    
    for msg in reversed(history):
        role = msg.get("role", "")
        if role == "user" and last_user is None:
            last_user = msg.get("content", "")
        elif role == "assistant" and last_assistant is None:
            last_assistant = msg.get("content", "")
        
        if last_user and last_assistant:
            break
    
    summary_parts = []
    if last_assistant:
        # Truncate if too long
        assistant_preview = last_assistant[:100] + "..." if len(last_assistant) > 100 else last_assistant
        summary_parts.append(f"Last assistant response: {assistant_preview}")
    
    if last_user:
        user_preview = last_user[:80] + "..." if len(last_user) > 80 else last_user
        summary_parts.append(f"Last user message: {user_preview}")
    
    if not summary_parts:
        return "Previous conversation context is available."
    
    return " | ".join(summary_parts)


def _local_cbt_therapist_reply(
    user_message: str,
    emotional_context: str,
    recent_history: List[Dict[str, str]],
) -> str:
    """
    Generate a multi-paragraph CBT-style reply using only local, deterministic logic.
    
    This function is fully local and makes no network calls. It uses string logic
    and heuristics to create a CBT-oriented therapist response.
    
    Args:
        user_message: Current user message
        emotional_context: Context string about emotional state and CBT suggestions
        recent_history: Recent conversation history
    
    Returns:
        Multi-paragraph CBT-style response string
    """
    # Check for crisis language first
    has_crisis = _check_crisis_language(user_message)
    
    # Start building response
    response_parts = []
    
    # Crisis disclaimer if needed
    if has_crisis:
        response_parts.append(_build_crisis_disclaimer())
    
    # Opening: Validation and empathy
    opening_phrases = [
        "Thank you for sharing this with me. It sounds like you're dealing with something really difficult, and it makes sense that you'd feel this way.",
        "I appreciate you opening up about this. What you're experiencing sounds challenging, and your feelings are completely valid.",
        "Thank you for trusting me with this. It takes courage to share what you're going through, and I want you to know I'm here to listen.",
    ]
    
    # Select opening based on message length/sentiment (simple heuristic)
    if len(user_message) < 50:
        opening = opening_phrases[2]  # Short message -> appreciation
    elif "feel" in user_message.lower() or "feeling" in user_message.lower():
        opening = opening_phrases[0]  # Emotional language -> validation
    else:
        opening = opening_phrases[1]  # Default -> appreciation
    
    response_parts.append(opening)
    response_parts.append("")
    
    # Reflection: Echo back key themes from user message
    # Extract key emotional words
    emotional_words = []
    common_emotions = [
        "sad", "angry", "anxious", "worried", "frustrated", "overwhelmed",
        "hopeless", "lonely", "stressed", "tired", "exhausted", "confused",
        "scared", "nervous", "disappointed", "hurt", "upset", "lost"
    ]
    
    user_lower = user_message.lower()
    for emotion in common_emotions:
        if emotion in user_lower:
            emotional_words.append(emotion)
    
    if emotional_words:
        unique_emotions = list(set(emotional_words))[:2]  # Max 2 unique emotions
        emotion_str = " and ".join(unique_emotions)
        reflection = (
            f"I'm hearing themes of {emotion_str} in what you're sharing. "
            "When we experience these kinds of feelings, it's natural for our minds to "
            "focus on the challenges we're facing."
        )
    else:
        reflection = (
            "From what you've shared, it sounds like you're navigating some real challenges. "
            "When we're in the midst of difficult situations, our thoughts can sometimes "
            "become very focused on what's hard, which can make things feel even more overwhelming."
        )
    
    response_parts.append(reflection)
    response_parts.append("")
    
    # Reference emotional context if available and meaningful
    if emotional_context and "No specific emotional context" not in emotional_context:
        context_note = (
            f"From the information available, {emotional_context.lower()}. "
            "This gives us some additional context to work with."
        )
        response_parts.append(context_note)
        response_parts.append("")
    
    # CBT-oriented questions and steps (2-5 bullet points)
    # Generate based on message content
    cbt_steps = []
    
    # Check for negative thought patterns
    if any(word in user_lower for word in ["always", "never", "nothing", "everything", "can't", "won't"]):
        cbt_steps.append(
            "**Thought examination**: I notice some absolute language in what you're sharing "
            "(words like 'always,' 'never,' 'everything'). When we're distressed, our minds "
            "often generate very absolute thoughts that *feel* completely true, but CBT invites us "
            "to gently question them. What would it be like to ask: 'Is this thought 100% true in "
            "every situation? What evidence supports it? What evidence challenges it?'"
        )
    
    # Check for isolation/loneliness themes
    if any(word in user_lower for word in ["alone", "lonely", "nobody", "no one", "isolated"]):
        cbt_steps.append(
            "**Connection exploration**: It sounds like there might be feelings of isolation or loneliness. "
            "Sometimes when we're struggling, we pull away from others, which can make things feel even harder. "
            "What would it be like to reach out to someone you trust—even if it's just for a brief conversation? "
            "Connection can be a powerful antidote to isolation."
        )
    
    # Check for overwhelm/stress
    if any(word in user_lower for word in ["overwhelmed", "too much", "can't handle", "too hard", "stress"]):
        cbt_steps.append(
            "**Breaking things down**: When things feel overwhelming, it can help to break them into "
            "smaller, more manageable pieces. What's one very small step you could take right now, "
            "or today, that would move you even slightly in a direction that feels better? "
            "Sometimes starting with the smallest possible action can create momentum."
        )
    
    # Check for past-focused or future-focused worry
    if any(word in user_lower for word in ["regret", "should have", "what if", "worry", "anxious"]):
        cbt_steps.append(
            "**Present-moment awareness**: Our minds have a tendency to travel to the past (with regret "
            "or 'should haves') or to the future (with worry and 'what ifs'). While these thoughts are "
            "natural, they can keep us stuck. What would it be like to gently bring your attention to "
            "this moment, right now? What's actually happening in this present moment that you can observe "
            "without judgment?"
        )
    
    # Default CBT steps if no specific patterns detected
    if not cbt_steps:
        cbt_steps = [
            "**Thought exploration**: When you notice a difficult or painful thought, try asking yourself: "
            "'What evidence supports this thought? What evidence challenges it? What would I tell a close "
            "friend who had this same thought?' Creating that small bit of distance between yourself and "
            "the thought can help you see it more clearly.",
            "**Behavioral activation**: Sometimes when we're struggling emotionally, we naturally pull back "
            "from activities. While that's understandable, sometimes doing even one small thing that aligns "
            "with our values or brings a moment of comfort can shift our experience. What's one tiny action "
            "you could take today—even something very small—that might help?"
        ]
    
    # Add at least 2 steps, up to 5
    if len(cbt_steps) < 2:
        cbt_steps.append(
            "**Gentle self-compassion**: Remember that you're human, and struggling doesn't mean you're "
            "failing. What would it be like to treat yourself with the same kindness you might offer to "
            "a friend going through something similar?"
        )
    
    # Limit to 5 steps
    cbt_steps = cbt_steps[:5]
    
    response_parts.append("Here are some reflections and steps you might consider:")
    response_parts.append("")
    for i, step in enumerate(cbt_steps, 1):
        response_parts.append(f"{i}. {step}")
        response_parts.append("")
    
    # Closing: Validation and encouragement
    closing = (
        "You don't have to figure everything out at once. Even noticing these patterns in your thoughts "
        "and feelings is already a meaningful step toward understanding and change. Take things one moment "
        "at a time, and be gentle with yourself in the process."
    )
    
    if not has_crisis:
        closing += (
            " If you ever feel overwhelmed, unsafe, or like you need more support than I can provide, "
            "please consider reaching out to a mental health professional in your area."
        )
    
    response_parts.append(closing)
    
    return "\n".join(response_parts)


def generate_psychologist_response(
    user_message: str,
    fused_vector: Optional[List[float]] = None,
    emotional_state: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    cbt_suggestions: Optional[List[str]] = None,
    config: Optional[PsychologistLLMConfig] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Generate a local, multi-turn CBT therapist response.
    
    This function is fully local and makes no external API calls. It uses deterministic
    string logic and heuristics to generate CBT-oriented therapist responses.
    
    Args:
        user_message: User's input message
        fused_vector: Optional fused multimodal embedding vector (presence noted, not printed)
        emotional_state: Optional human-readable emotional state description
        history: Optional list of previous messages (each dict with 'role' and 'content' keys)
        cbt_suggestions: Optional list of CBT-style suggestion strings
        config: Optional configuration for therapist parameters
    
    Returns:
        Tuple of (assistant_reply: str, updated_history: List[Dict[str, str]])
        - assistant_reply: The generated therapist response
        - updated_history: History with new user message and assistant reply appended
    """
    # Initialize config
    if config is None:
        config = PsychologistLLMConfig()
    
    # Normalize history to a list
    if history is None:
        history = []
    
    # Clip history to maximum size before processing
    history = _clip_history(history, max_turns=config.max_history_turns)
    
    # Build emotional context string
    emotional_context = _build_emotional_context(
        emotional_state=emotional_state,
        cbt_suggestions=cbt_suggestions,
        fused_vector=fused_vector,
    )
    
    # Get recent history summary (for context, though we pass full recent_history to the function)
    recent_history = history.copy()
    
    # Generate response using local CBT therapist
    assistant_reply = _local_cbt_therapist_reply(
        user_message=user_message,
        emotional_context=emotional_context,
        recent_history=recent_history,
    )
    
    # Append new messages to history
    history.append({
        "role": "user",
        "content": user_message
    })
    history.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    # Clip history again to maximum size
    history = _clip_history(history, max_turns=config.max_history_turns)
    
    return assistant_reply, history
