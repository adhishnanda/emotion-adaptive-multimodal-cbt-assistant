"""Rule-based CBT response generator."""

from typing import Any, Dict, List, Optional, Tuple


def handle_sadness(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for sadness.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "It makes sense that you're feeling sad right now. Sadness is a natural response "
        "to difficult experiences, and it's okay to acknowledge these feelings.\n\n"
        "Sometimes when we feel sad, our thoughts can become more negative or we might "
        "withdraw from activities we usually enjoy. Let's gently explore what might be "
        "contributing to these feelings.\n\n"
        "Can you identify any specific thoughts or situations that are connected to "
        "this sadness? Sometimes writing down what we're thinking can help us see "
        "patterns and find new perspectives."
    )

    steps = [
        "Take a moment to identify the specific situation or thought that triggered this sadness.",
        "Consider: Is there another way to look at this situation? What would you tell a friend in a similar situation?",
        "Think of one small activity you could do today that might bring a moment of comfort or connection.",
        "Practice self-compassion: Remind yourself that feeling sad doesn't mean you're weak or failing.",
    ]

    return {"response": response, "steps": steps}


def handle_anger(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for anger.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "Anger is a powerful emotion, and it's completely valid to feel this way. "
        "It often signals that something important to us feels threatened or unfair.\n\n"
        "When we're angry, our thoughts can become intense and our bodies can feel "
        "activated. Let's take a moment to understand what's beneath the anger.\n\n"
        "Anger often masks other feelings like hurt, fear, or frustration. Can you "
        "identify what need or value feels threatened right now? Understanding the "
        "root of anger can help us respond in ways that align with our values."
    )

    steps = [
        "Take a few deep breaths to help your body calm down before responding.",
        "Identify the specific trigger: What exactly happened that made you feel angry?",
        "Explore the underlying need: What value or need feels threatened? (e.g., respect, fairness, safety)",
        "Consider your response options: What actions would align with your values and help address the situation constructively?",
    ]

    return {"response": response, "steps": steps}


def handle_fear(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for fear.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "Feeling afraid or anxious is your mind's way of trying to protect you. "
        "These feelings, while uncomfortable, are a normal part of being human.\n\n"
        "When we feel fear, our thoughts often focus on worst-case scenarios. "
        "Let's gently examine whether the feared outcome is as likely or as severe "
        "as it might feel right now.\n\n"
        "Sometimes grounding techniques can help us feel more present and less "
        "overwhelmed by fear. Can you identify what specifically you're afraid of, "
        "and what evidence you have about whether that fear will come true?"
    )

    steps = [
        "Practice grounding: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "Challenge catastrophic thinking: What's the worst that could happen? What's most likely to happen? What's the best outcome?",
        "Identify what you can control: Focus on actions you can take rather than worrying about things outside your control.",
        "Consider gradual exposure: If the fear is about something you can safely face, think about taking small steps toward it.",
    ]

    return {"response": response, "steps": steps}


def handle_joy(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for joy.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "It's wonderful that you're experiencing joy! Positive emotions are just as "
        "important to acknowledge and explore as difficult ones.\n\n"
        "When we feel joy, it can be helpful to notice what thoughts, activities, or "
        "connections are contributing to this positive feeling. Understanding what "
        "brings us joy can help us create more opportunities for these experiences.\n\n"
        "Consider savoring this moment and reflecting on what made it possible. "
        "Sometimes we can learn valuable lessons about our values and needs from "
        "our positive experiences."
    )

    steps = [
        "Savor the moment: Take time to fully experience and appreciate this feeling of joy.",
        "Identify the source: What specific thoughts, activities, or connections contributed to this joy?",
        "Consider how to create more opportunities: What can you do to invite more of these positive experiences?",
        "Practice gratitude: Reflect on what you're grateful for in this moment.",
    ]

    return {"response": response, "steps": steps}


def handle_disgust(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for disgust.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "Disgust is a strong emotional response that often signals something feels "
        "wrong, harmful, or against our values. It's valid to feel this way.\n\n"
        "When we feel disgust, it can be helpful to identify what specifically is "
        "triggering this response. Is it a situation, a behavior, or perhaps a "
        "violation of your values?\n\n"
        "Understanding what's behind the disgust can help us determine whether it's "
        "a signal to set boundaries, take action, or perhaps examine our expectations. "
        "What feels most important to address here?"
    )

    steps = [
        "Identify the trigger: What specifically is causing this feeling of disgust?",
        "Examine your values: What value or boundary feels violated?",
        "Consider your response: What action (if any) would align with your values?",
        "Practice self-compassion: Remember that having strong reactions doesn't make you wrong or bad.",
    ]

    return {"response": response, "steps": steps}


def handle_surprise(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for surprise.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "Surprise can be a complex emotion—it might feel positive, negative, or "
        "somewhere in between. It's natural to feel uncertain when something unexpected happens.\n\n"
        "When we're surprised, our minds often need a moment to process and make sense "
        "of what's happening. It's okay to take time to understand your feelings about "
        "this unexpected situation.\n\n"
        "Consider: How do you feel about this surprise? What does it mean to you? "
        "Sometimes surprises can be opportunities for growth, even when they're initially "
        "uncomfortable."
    )

    steps = [
        "Take a moment to process: Give yourself time to understand what happened and how you feel about it.",
        "Assess the impact: Is this surprise positive, negative, or neutral? How does it affect your goals or values?",
        "Consider your response: What actions would be most helpful in responding to this surprise?",
        "Look for opportunities: Even difficult surprises can sometimes lead to growth or new possibilities.",
    ]

    return {"response": response, "steps": steps}


def handle_neutral(user_text: str) -> Dict[str, Any]:
    """
    Generate CBT response for neutral emotion.

    Args:
        user_text: User's input text

    Returns:
        Dictionary with response components
    """
    response = (
        "It sounds like you're in a relatively neutral or balanced emotional state right now. "
        "This can be a good time for reflection and planning.\n\n"
        "Neutral moments are valuable opportunities to check in with yourself: How are you "
        "doing overall? What's working well in your life? What might you want to adjust?\n\n"
        "Consider using this time to practice mindfulness, set intentions, or engage in "
        "activities that support your wellbeing. Sometimes maintaining balance requires "
        "active attention to our needs."
    )

    steps = [
        "Practice mindfulness: Take a few moments to notice your thoughts, feelings, and physical sensations without judgment.",
        "Reflect on your needs: What do you need right now to maintain or improve your wellbeing?",
        "Consider your values: Are you living in ways that align with what matters most to you?",
        "Plan for balance: What small actions can you take to support your emotional and mental health?",
    ]

    return {"response": response, "steps": steps}


def adjust_for_depression_risk(
    base_response: str, depression_risk: Optional[float]
) -> Tuple[str, Optional[str]]:
    """
    Adjust response tone and add safety message based on depression risk.

    Args:
        base_response: Base response text
        depression_risk: Depression risk score (0-1) or None

    Returns:
        Tuple of (adjusted_response, safety_message)
    """
    if depression_risk is None:
        return base_response, None

    safety_message = None

    if depression_risk < 0.3:
        # No change needed
        return base_response, None

    elif 0.3 <= depression_risk < 0.7:
        # Soften tone
        softened_intro = (
            "Thanks for sharing this with me. I can hear that you're going through "
            "something difficult. Let's explore this gently together.\n\n"
        )
        adjusted_response = softened_intro + base_response
        return adjusted_response, None

    else:  # depression_risk >= 0.7
        # Increase validation and add safety message
        validated_intro = (
            "Thank you for trusting me with these feelings. What you're experiencing "
            "sounds really difficult, and it takes courage to acknowledge and share "
            "these emotions. Your feelings are valid and important.\n\n"
        )
        adjusted_response = validated_intro + base_response

        safety_message = (
            "This isn't a medical diagnosis, but the feelings you're describing sound difficult. "
            "If you ever feel overwhelmed, lonely, or unsafe, please consider reaching out to a "
            "mental health professional or crisis resource near you. You don't have to face "
            "these challenges alone, and there are people who want to help."
        )

        return adjusted_response, safety_message


def generate_cbt_response(
    user_text: str,
    emotion_label: str,
    depression_risk: Optional[float] = None,
    extra_signals: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generates a structured CBT-style response combining:
    - empathic reflection
    - emotion-specific guidance
    - depression-risk-adjusted tone
    - optional safety notes

    Args:
        user_text: User's input text
        emotion_label: Detected emotion label (e.g., "anger", "sadness", "joy")
        depression_risk: Depression risk probability (0-1) or None
        extra_signals: Additional signals for future expansion

    Returns:
        Dictionary with:
            - emotion_label: str
            - depression_risk: float or None
            - response: str (combined response text)
            - steps: List[str] (CBT-style prompts)
            - safety: str or None (safety message if applicable)
    """
    # Normalize emotion label to lowercase
    emotion_label = emotion_label.lower().strip()

    # Dispatch table for emotion handlers
    emotion_handlers = {
        "sadness": handle_sadness,
        "anger": handle_anger,
        "fear": handle_fear,
        "joy": handle_joy,
        "disgust": handle_disgust,
        "surprise": handle_surprise,
        "neutral": handle_neutral,
    }

    # Get emotion-specific response
    handler = emotion_handlers.get(emotion_label, handle_neutral)
    emotion_data = handler(user_text)

    base_response = emotion_data["response"]
    steps = emotion_data.get("steps", [])

    # Adjust for depression risk
    adjusted_response, safety_message = adjust_for_depression_risk(
        base_response, depression_risk
    )

    return {
        "emotion_label": emotion_label,
        "depression_risk": depression_risk,
        "response": adjusted_response,
        "steps": steps,
        "safety": safety_message,
    }
