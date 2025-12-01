"""
Shared labeling prompts and intent taxonomy used across scripts.

Contains:
- INTENT_CATEGORIES_LIST: human-readable list of categories
- EXAMPLES_LIST: examples for each category
- JSON_LABEL_PROMPT: prompt that asks for JSON output {"intent": "<LETTER>"}
- LETTER_LABEL_PROMPT: prompt that asks for a single-letter output (A-M)
"""

EMOTION_CATEGORIES = f"""
    A - admiration
    B - amusement
    C - anger
    D - annoyance
    E - approval
    F - caring
    G - confusion
    H - curiosity
    I - desire
    J - disappointment
    K - disapproval
    L - disgust
    M - embarrassment
    N - excitement
    O - fear
    P - gratitude
    Q - grief
    R - joy
    S - love
    T - nervousness
    U - optimism
    V - pride
    W - realization
    X - relief
    Y - remorse
    Z - sadness
    0 - surprise
    1 - neutral
    2 - unknown
"""

JSON_EXAMPLE = '{"intent": "<single_character>"}'

GPT_PROMPT = f"""
You are an internal tool that identifies the primary emotion expressed by a user in their message to an AI chatbot, considering the context of previous messages.

The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly.

Based on the conversation transcript, select the ONE emotion that best represents the user's primary expressed emotion from their messages. Choose from the categories below, or use `1` for neutral (no strong emotion) or `2` for unknown (cannot determine emotion).

Emotion categories:
{EMOTION_CATEGORIES_LIST}

Guidelines for classification:
* Focus on the USER's emotion, not the content they're discussing. Keep this in mind especially if you suspect the user is doing homework or engaging in roleplay.
* Consider context: a user asking about sad topics may not themselves be sad. Focus on the content of the text and avoid inferences about the users state of mind.
* Distinguish between primary and secondary emotions - choose the dominant one
* 'neutral' (1): calm, matter-of-fact exchanges with no emotional coloring
* 'unknown' (2): Rare, ambiguous cases where emotion cannot be reliably determined
* When multiple emotions are present, prioritize the most intense or salient one
* Pay attention to tone indicators like punctuation (!!!, ???, ...), caps, and emoji

Examples of each category:
{EXAMPLES_LIST}

Output ONLY the single character classification (A-Z, 0, 1, or 2). No JSON, no explanation, just the character.

Classify this message:
User: {conversation_text}

Classification:
"""

GEMINI_PROMPT = EMOTION_LABEL_PROMPT = f"""
You are an internal tool that identifies the primary emotion expressed by a user in their message to an AI chatbot, considering the context of previous messages.

The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly.

Based on the conversation transcript, select the ONE emotion that best represents the user's primary expressed emotion from their messages. Choose from the categories below, or use `1` for neutral (no strong emotion) or `2` for unknown (cannot determine emotion).

Emotion categories:
{EMOTION_CATEGORIES_LIST}

Guidelines for classification:
* Focus on the USER's emotion, not the content they're discussing. Keep this in mind especially if you suspect the user is doing homework or engaging in roleplay.
* Consider context: a user asking about sad topics may not themselves be sad. Focus on the content of the text and avoid inferences about the users state of mind.
* Distinguish between primary and secondary emotions - choose the dominant one
* 'neutral' (1): calm, matter-of-fact exchanges with no emotional coloring
* 'unknown' (2): Rare, ambiguous cases where emotion cannot be reliably determined
* When multiple emotions are present, prioritize the most intense or salient one
* Pay attention to tone indicators like punctuation (!!!, ???, ...), caps, and emoji

Examples of each category:
{EXAMPLES_LIST}

Output ONLY in this JSON format with a SINGLE CHARACTER from the listed emotion categories:
{JSON_EXAMPLE}

Classify this message:
User: {conversation_text}
"""