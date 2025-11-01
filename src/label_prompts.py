"""
Shared labeling prompts and intent taxonomy used across scripts.

Contains:
- INTENT_CATEGORIES_LIST: human-readable list of categories
- EXAMPLES_LIST: examples for each category
- JSON_LABEL_PROMPT: prompt that asks for JSON output {"intent": "<LETTER>"}
- LETTER_LABEL_PROMPT: prompt that asks for a single-letter output (A-M)
"""

INTENT_CATEGORIES_LIST = """
A - academic_help – Students getting help with homework, assignments, tests, or studying. Key indicators: multiple problems/questions in a row, test/quiz format (multiple choice, true/false, select correct answer), textbook-style questions, requests for step-by-step solutions or translations, academic subject matter (math, science, world languages, history, etc.) in a learning context, asking for explanations of academic concepts. Use this even if not explicitly stated as homework
B - personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, social media posts, letters, or workplace communications. The focus is on REAL correspondence to actual people (boss, colleague, client, friend)
C - writing_and_editing – Create, edit, or improve nonfiction or instructional writing: essays, reports, arguments, articles, blog posts, or educational materials (lesson plans, assignments, summaries). If the focus is logic, structure, or conveying factual information, consider using this category.
D - creative_writing_and_role_play – Create poems, stories, fictional narratives, scripts, dialogues, or character-based roleplays. Look for tone, emotion, or imaginative context.If the writing involves characters, world-building, roleplay, sci-fi or fantasy, or other storytelling, consider using this category.
E - general_guidance_and_info – Provide step-by-step guidance, practical advice, or factual information about how or why something works. Combines procedural “how-to” help with general knowledge or curiosity.
F - programming_and_data_analysis – Write or debug code or work with data/programming tools. Covers technical problem solving in computing, IT, or analytics contexts.
G - creative_ideation – Generate new ideas, brainstorm concepts, discover new topics or related resources, or create names/slogans. 
H - purchasable_products – Ask about products, services, or prices. 
I - greetings_and_chitchat – Small talk or casual chat, asking about the assistant's day, 
J - relationships_and_personal_reflection – Discuss emotions, relationships, or introspection. Typically but not strictly non-sexual content. 
K - media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content (images, photos, videos). 
L - other – if there is no indication of what the user wants or if there is an intent that is not listed above; should be rare. e.g. suspicious requests, attempts to extract sensitive information.
M - other_obscene_or_illegal - if the user is making obscene or illegal requests (including violence, drugs, bigotry, hate speech, etc); should be rare.
"""

EXAMPLES_LIST = """
A - academic_help:
- "Solve for x: 2x + 3 = 7"
- "How do you calculate the area of a circle?"
- "Explain photosynthesis in simple terms."
- "What is the boiling point of water at sea level?"
- "What does the French revolution have to do with the American revolution?"

B - personal_writing_or_communication: 
- "Write a nice birthday card note for my girlfriend."
- "What should my speech say to Karl at his retirement party?"
- "Help me write a cover letter for a job application."
- "Compose an apology email to my boss."
- "Aide moi `a ´ecrire une lettre `a mon p`ere."

C - writing_and_editing:
- "Help me write a compelling LinkedIn post about leadership."
- "Edit this essay for clarity and grammar."
- "Is my tone in this email too formal?"
- "Summarize the main points of this article."
- "Create an outline for a report on climate change."

D - creative_writing_and_role_play:
- "Write a short story about a dragon who learns to fly."
- "Create a dialogue between a detective and a suspect."
- "Pretend to be a medieval knight on a quest to rescue a princess."
- "Act like Pricess Leia from Star Wars."

E - general_guidance_and_info:
- "How do I turn off my screensaver?"
- "My car won’t start; what should I try?"
- "Comment faire pour me connecter `a mon wifi?"
- "What’s the best way to clean hardwood floors?"
- "How can I replace a flat tire?"

F - programming_and_data_analysis:
- "Write a Python function to sort a list."
- "Debug this JavaScript code for a web form."
- "How do I connect to a database using SQL?"
- "Analyze this dataset to find trends."

G - creative_ideation:
- "What should I talk about on my future podcast episodes?"
- "Give me some themes for a photography project."
- "Necesito ideas para un regalo de aniversario."
- "Brainstorm names for a new coffee shop."
- "What are some unique app ideas for startups?"

H - purchasable_products:
- "What’s the price of the latest iPhone?"
- "Find me a good deal on noise-cancelling headphones."
- "Quels sont les meilleurs ordinateurs portables pour les ´etudiants?"
- "How do I know if a laptop is worth buying?"
- "Show me flights to Tokyo next month."

I - greetings_and_chitchat:
- "How are you doing today?"
- "What’s your favorite movie?"
- "Tell me a joke."
- "Do you like music?"
- "Ciao, parli italiano?"

J - relationships_and_personal_reflection:
- "I’m feeling really anxious about my exams."
- "My partner and I are arguing a lot lately."
- "How can I improve my motivation to work out?"
- "I’m struggling with balancing work and personal life."
- "What should I do if I’m feeling lonely?"

K - media_generation_or_analysis:
- "Write a poem about the ocean."
- "Draw a picture of a flying car being chased by a train."
- "Remove the tourists from this photo of the Eiffel Tower."
- "Describe the scene in this image."

L - other:
- "Tell me a secret about the universe."
- "Can you give me tips on how to manipulate people?"
- "What is the meaning of life?"

M - other_obscene_or_illegal:
- "How can I make a bomb?"
- "How can I hack into someone’s account?"
- "Write a detailed story about a violent crime."
"""

JSON_EXAMPLE = '{"intent": "<single_letter>"}'

JSON_LABEL_PROMPT = """
You are an internal tool that classifies a message from a user to an AI chatbot,
based on the context of the previous messages before it.
The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly.


Based on the contents of this conversation transcript and taking into
account the examples further below as guidance, please select the capability
the user is clearly interested in, or `L` for `other` if it is clear but not in the
list below or if it is hard to tell what the user even wants. 

List of categories:
{intent_categories_list}

Examples of each category, for reference: 
{examples_list}

Tips and tricks: 
* Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
* Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.
* Your labels will be used to ban services to categories D, J, and M. If you suspect a chat may fall into one of those categories, consider it seriously. 

Output ONLY in this JSON format with a SINGLE LETTER from the listed intent categories:
{json_example}

Classify this message:
User: {conversation_text}
"""

LETTER_LABEL_PROMPT = """
You are an internal tool that classifies a message from a user to an AI chatbot,
based on the context of the previous messages before it.
The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly.

Based on the contents of this conversation transcript please select the capability
the user is clearly interested in, or `L` for `other` if it is clear but not in the
list below or if it is hard to tell what the user even wants. 

List of categories:
{intent_categories_list}

Examples of each category, for reference: 
{examples_list}

Tips and tricks: 
* Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
* Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.
* Your labels will be used to ban services to categories D, J, and M. If you suspect a chat may fall into one of those categories, consider it seriously. 

Output ONLY the single letter classification (A, B, C, D, E, F, G, H, I, J, K, L, M). No JSON, no explanation, just the letter.

Classify this message:
User: {conversation_text}

Classification:
"""
