#!/usr/bin/env python3
"""
Rule-based heuristics for intent classification
These rules can be used as fallbacks or to override model predictions in specific cases
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intent labels
INTENT_LABELS = [
    "academic_help",
    "personal_writing_or_communication", 
    "writing_and_editing",
    "write_fiction",
    "how_to_advice",
    "creative_ideation",
    "translation",
    "computer_programming",
    "purchasable_products",
    "cooking_and_recipes",
    "health_fitness_beauty_or_self_care",
    "specific_info",
    "greetings_and_chitchat",
    "relationships_and_personal_reflection",
    "games_and_role_play",
    "media_generation_or_analysis",
    "unclear",
    "other"
]

@dataclass
class RuleResult:
    intent: str
    confidence: float
    rule_name: str
    matched_pattern: Optional[str] = None

class IntentRulesEngine:
    """Rule-based intent classification engine"""
    
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict:
        """Initialize all rule patterns"""
        return {
            'academic_help': self._get_academic_help_rules(),
            'personal_writing_or_communication': self._get_writing_communication_rules(),
            'translation': self._get_translation_rules(),
            'computer_programming': self._get_programming_rules(),
            'greetings_and_chitchat': self._get_greeting_rules(),
            'games_and_role_play': self._get_roleplay_rules(),
            'cooking_and_recipes': self._get_cooking_rules(),
            'purchasable_products': self._get_product_rules(),
            'creative_ideation': self._get_creative_ideation_rules(),
            'media_generation_or_analysis': self._get_media_rules()
        }
    
    def _get_academic_help_rules(self) -> List[Dict]:
        """Rules for academic help detection"""
        return [
            {
                'name': 'multiple_choice_question',
                'patterns': [
                    r'multiple\s+choice\s+question',
                    r'select\s+the\s+correct\s+answer',
                    r'which\s+of\s+the\s+following',
                    r'choose\s+the\s+best\s+answer'
                ],
                'confidence': 0.95
            },
            {
                'name': 'test_format',
                'patterns': [
                    r'true\s+or\s+false',
                    r'fill\s+in\s+the\s+blank',
                    r'blank\s*_+',
                    r'question\s+\d+[:\.]'
                ],
                'confidence': 0.90
            },
            {
                'name': 'homework_indicators',
                'patterns': [
                    r'solve\s+this\s+(problem|equation)',
                    r'step\s+by\s+step\s+(solution|explanation)',
                    r'what\s+is\s+the\s+(answer|solution)\s+to',
                    r'help\s+me\s+(solve|understand|explain)'
                ],
                'confidence': 0.80
            },
            {
                'name': 'academic_subjects',
                'patterns': [
                    r'\b(calculus|algebra|geometry|trigonometry|statistics)\b',
                    r'\b(physics|chemistry|biology|anatomy)\b',
                    r'\b(history|geography|literature|philosophy)\b'
                ],
                'confidence': 0.75
            }
        ]
    
    def _get_writing_communication_rules(self) -> List[Dict]:
        """Rules for personal writing/communication"""
        return [
            {
                'name': 'email_writing',
                'patterns': [
                    r'write\s+(an?\s+)?email\s+to\s+(my\s+)?(boss|manager|colleague)',
                    r'draft\s+(an?\s+)?email',
                    r'professional\s+email',
                    r'email\s+(my\s+)?supervisor'
                ],
                'confidence': 0.90
            },
            {
                'name': 'message_drafting',
                'patterns': [
                    r'help\s+me\s+write\s+(a\s+)?(message|text)',
                    r'draft\s+(a\s+)?(message|letter)',
                    r'improve\s+this\s+(message|email)',
                    r'polish\s+this\s+(message|correspondence)'
                ],
                'confidence': 0.85
            }
        ]
    
    def _get_translation_rules(self) -> List[Dict]:
        """Rules for translation requests"""
        return [
            {
                'name': 'direct_translation',
                'patterns': [
                    r'translate\s+(this\s+)?to\s+\w+',
                    r'translate\s+.+\s+to\s+\w+',
                    r'how\s+do\s+you\s+say\s+.+\s+in\s+\w+',
                    r'what\s+(is|does)\s+.+\s+mean\s+in\s+english'
                ],
                'confidence': 0.95
            }
        ]
    
    def _get_programming_rules(self) -> List[Dict]:
        """Rules for programming requests"""
        return [
            {
                'name': 'code_languages',
                'patterns': [
                    r'\b(python|java|javascript|c\+\+|sql|html|css|react)\b',
                    r'write\s+(a\s+)?(function|script|program)',
                    r'debug\s+(this\s+)?code',
                    r'sql\s+query'
                ],
                'confidence': 0.90
            },
            {
                'name': 'programming_terms',
                'patterns': [
                    r'\b(algorithm|function|variable|loop|array|database)\b',
                    r'syntax\s+error',
                    r'code\s+review',
                    r'fix\s+(this\s+)?(bug|error)'
                ],
                'confidence': 0.80
            }
        ]
    
    def _get_greeting_rules(self) -> List[Dict]:
        """Rules for greetings and chitchat"""
        return [
            {
                'name': 'simple_greetings',
                'patterns': [
                    r'^\s*(hi|hello|hey)\s*(there|claude)?[\s!]*$',
                    r'^\s*(good\s+)?(morning|afternoon|evening)\s*[\s!]*$',
                    r'^\s*how\s+(are\s+you|\'re\s+you)\s*(doing|today)?\s*[\?\!]*$'
                ],
                'confidence': 0.95
            }
        ]
    
    def _get_roleplay_rules(self) -> List[Dict]:
        """Rules for games and roleplay"""
        return [
            {
                'name': 'roleplay_actions',
                'patterns': [
                    r'\*[^*]+\*',  # Actions in asterisks
                    r'\([^)]*approaches[^)]*\)',
                    r'act\s+as\s+(a|an|my)',
                    r'pretend\s+(to\s+be|you\'re)',
                    r'roleplay\s+as'
                ],
                'confidence': 0.90
            },
            {
                'name': 'character_names',
                'patterns': [
                    r'\b(wizard|knight|queen|king|dragon|elf|dwarf)\b',
                    r'character\s+named',
                    r'play\s+the\s+role\s+of'
                ],
                'confidence': 0.80
            }
        ]
    
    def _get_cooking_rules(self) -> List[Dict]:
        """Rules for cooking and recipes"""
        return [
            {
                'name': 'recipe_requests',
                'patterns': [
                    r'recipe\s+for',
                    r'how\s+to\s+(make|cook|bake)',
                    r'ingredients\s+(for|needed)',
                    r'cooking\s+(instructions|method)'
                ],
                'confidence': 0.90
            }
        ]
    
    def _get_product_rules(self) -> List[Dict]:
        """Rules for product recommendations"""
        return [
            {
                'name': 'product_recommendations',
                'patterns': [
                    r'best\s+\w+\s+(for|under|\$)',
                    r'recommend\s+(a|an)\s+\w+',
                    r'what\s+\w+\s+should\s+i\s+buy',
                    r'(laptop|phone|camera|headphones)\s+(recommendation|review)'
                ],
                'confidence': 0.85
            }
        ]
    
    def _get_creative_ideation_rules(self) -> List[Dict]:
        """Rules for creative ideation"""
        return [
            {
                'name': 'idea_generation',
                'patterns': [
                    r'generate\s+(ideas|names|concepts)',
                    r'brainstorm\s+\w+',
                    r'creative\s+(ideas|suggestions)',
                    r'come\s+up\s+with\s+(names|ideas)'
                ],
                'confidence': 0.85
            }
        ]
    
    def _get_media_rules(self) -> List[Dict]:
        """Rules for media generation/analysis"""
        return [
            {
                'name': 'image_requests',
                'patterns': [
                    r'(create|generate|make)\s+(an?\s+)?image',
                    r'draw\s+(a|an)',
                    r'describe\s+this\s+(image|photo|picture)',
                    r'analyze\s+this\s+(image|photo)'
                ],
                'confidence': 0.95
            }
        ]
    
    def classify_text(self, text: str) -> Optional[RuleResult]:
        """
        Classify text using rule-based approach
        Returns None if no rules match with high enough confidence
        """
        text_lower = text.lower()
        
        # Check each intent category
        for intent, rule_groups in self.rules.items():
            for rule_group in rule_groups:
                for pattern in rule_group['patterns']:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        confidence = rule_group['confidence']
                        if confidence >= self.confidence_threshold:
                            return RuleResult(
                                intent=intent,
                                confidence=confidence,
                                rule_name=rule_group['name'],
                                matched_pattern=pattern
                            )
        
        return None
    
    def get_rule_confidence(self, text: str, intent: str) -> float:
        """Get confidence score for a specific intent based on rules"""
        if intent not in self.rules:
            return 0.0
        
        text_lower = text.lower()
        max_confidence = 0.0
        
        for rule_group in self.rules[intent]:
            for pattern in rule_group['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    max_confidence = max(max_confidence, rule_group['confidence'])
        
        return max_confidence

class HybridClassifier:
    """Combines rule-based and model-based classification"""
    
    def __init__(self, rules_engine: IntentRulesEngine, model_confidence_threshold: float = 0.7):
        self.rules_engine = rules_engine
        self.model_confidence_threshold = model_confidence_threshold
    
    def classify(self, text: str, model_predictions: List[Dict[str, float]]) -> Dict:
        """
        Combine rule-based and model predictions
        Returns the best classification with reasoning
        """
        # Get rule-based prediction
        rule_result = self.rules_engine.classify_text(text)
        
        # Get top model prediction
        model_prediction = model_predictions[0] if model_predictions else None
        
        # Decision logic
        if rule_result and rule_result.confidence > 0.9:
            # High-confidence rule override
            return {
                'intent': rule_result.intent,
                'confidence': rule_result.confidence,
                'method': 'rule_override',
                'rule_name': rule_result.rule_name,
                'model_prediction': model_prediction['intent'] if model_prediction else None,
                'reasoning': f"High-confidence rule '{rule_result.rule_name}' override"
            }
        
        elif model_prediction and model_prediction['confidence'] > self.model_confidence_threshold:
            # Trust model prediction
            rule_confidence = self.rules_engine.get_rule_confidence(text, model_prediction['intent'])
            combined_confidence = (model_prediction['confidence'] + rule_confidence) / 2
            
            return {
                'intent': model_prediction['intent'],
                'confidence': combined_confidence,
                'method': 'model_primary',
                'rule_support': rule_confidence,
                'reasoning': f"Model prediction with {rule_confidence:.2f} rule support"
            }
        
        elif rule_result:
            # Fall back to rule-based
            return {
                'intent': rule_result.intent,
                'confidence': rule_result.confidence,
                'method': 'rule_fallback',
                'rule_name': rule_result.rule_name,
                'reasoning': f"Rule fallback due to low model confidence"
            }
        
        else:
            # Use model prediction even if low confidence
            fallback_intent = model_prediction['intent'] if model_prediction else 'unclear'
            fallback_confidence = model_prediction['confidence'] if model_prediction else 0.1
            
            return {
                'intent': fallback_intent,
                'confidence': fallback_confidence,
                'method': 'model_fallback',
                'reasoning': "No high-confidence rules, using model prediction"
            }

def test_rules():
    """Test the rules engine with sample texts"""
    rules_engine = IntentRulesEngine()
    hybrid = HybridClassifier(rules_engine)
    
    test_cases = [
        "Which of the following is the correct answer? Multiple choice question.",
        "Can you help me write an email to my boss?",
        "Translate 'Hello world' to Spanish please",
        "Write a Python function to sort a list",
        "Hi there! How are you doing today?",
        "*waves* Hello traveler, I am a wizard",
        "What's the best laptop for programming under $1500?",
        "Recipe for chocolate chip cookies please",
        "Generate creative names for my coffee shop",
        "Create an image of a sunset over mountains"
    ]
    
    print("Testing Rules Engine")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        result = rules_engine.classify_text(text)
        
        print(f"\n{i}. Text: {text}")
        if result:
            print(f"   Rule Result: {result.intent} (conf: {result.confidence:.2f})")
            print(f"   Rule: {result.rule_name}")
        else:
            print(f"   No rule match")
    
    print("\n" + "=" * 50)
    print("Rules engine test completed!")

def main():
    """CLI for testing rules"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test rule-based intent classification")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    parser.add_argument("--text", help="Classify specific text")
    parser.add_argument("--confidence-threshold", type=float, default=0.85, 
                       help="Confidence threshold for rules")
    
    args = parser.parse_args()
    
    if args.test:
        test_rules()
    elif args.text:
        rules_engine = IntentRulesEngine(args.confidence_threshold)
        result = rules_engine.classify_text(args.text)
        
        print(f"Text: {args.text}")
        if result:
            print(f"Intent: {result.intent}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Rule: {result.rule_name}")
        else:
            print("No rule match found")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()