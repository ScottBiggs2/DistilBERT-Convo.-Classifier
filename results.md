# Run 1 - Bad labels (20k), alpha = 0.7, 5 epochs:

Validation: 

```
2025-10-29 07:09:30,235 - INFO - 
📋 VALIDATION CLASSIFICATION REPORT
2025-10-29 07:09:30,235 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.70      0.82      0.76       704
           B       0.74      0.81      0.77       399
           C       0.52      0.27      0.36       165
           D       0.76      0.75      0.76       530
           E       0.53      0.58      0.56       398
           F       0.88      0.62      0.73        80
           G       0.76      0.58      0.66        33
           H       1.00      0.03      0.07        59
           I       0.81      0.63      0.71       125
           J       0.62      0.64      0.63       202
           K       0.61      0.66      0.63        82
           L       0.27      0.30      0.28       185
           M       0.53      0.22      0.31        37

    accuracy                           0.66      2999
   macro avg       0.67      0.53      0.55      2999
weighted avg       0.66      0.66      0.65      2999

2025-10-29 07:09:30,682 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled/validation_confusion_matrix.png
2025-10-29 07:09:30,683 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-10-29 07:09:30,683 - INFO -    Total predictions: 2999
2025-10-29 07:09:30,683 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 07:09:30,683 - INFO -    Cross-category error rate: 0.00%
2025-10-29 07:09:30,683 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 07:09:30,683 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 07:09:30,683 - INFO -    Overall accuracy: 65.66%
2025-10-29 07:09:30,683 - INFO - 💾 Validation results saved to models/distilbert_distilled/validation_results.json
2025-10-29 07:09:30,683 - INFO - 🧪 Evaluating on held-out test set...
2025-10-29 07:09:30,683 - INFO - 📊 Running comprehensive evaluation on test set...
```

And Test:

```
2025-10-29 07:10:27,319 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-10-29 07:10:27,319 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.68      0.81      0.74       699
           B       0.68      0.76      0.72       386
           C       0.44      0.21      0.28       160
           D       0.74      0.76      0.75       549
           E       0.52      0.58      0.55       389
           F       0.90      0.69      0.78        90
           G       0.67      0.35      0.46        57
           H       1.00      0.09      0.17        66
           I       0.89      0.50      0.64       113
           J       0.65      0.65      0.65       203
           K       0.53      0.55      0.54        64
           L       0.25      0.30      0.27       170
           M       0.68      0.24      0.35        55

    accuracy                           0.64      3001
   macro avg       0.66      0.50      0.53      3001
weighted avg       0.65      0.64      0.62      3001

2025-10-29 07:10:27,756 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-10-29 07:10:27,757 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-10-29 07:10:27,757 - INFO -    Total predictions: 3001
2025-10-29 07:10:27,757 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 07:10:27,757 - INFO -    Cross-category error rate: 0.00%
2025-10-29 07:10:27,757 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 07:10:27,757 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 07:10:27,757 - INFO -    Overall accuracy: 63.61%
2025-10-29 07:10:27,757 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json
2025-10-29 07:10:27,757 - INFO - 
============================================================
2025-10-29 07:10:27,757 - INFO - TRAINING COMPLETION SUMMARY
2025-10-29 07:10:27,757 - INFO - ============================================================
2025-10-29 07:10:27,757 - INFO - 🎯 Validation Accuracy: 0.657
2025-10-29 07:10:27,757 - INFO - 🧪 Test Accuracy: 0.636
2025-10-29 07:10:27,757 - INFO - 📉 Accuracy Drop: 0.020 (✅ OK)
2025-10-29 07:10:27,757 - INFO - 🚨 Validation Cross-Category Errors: 0.000
2025-10-29 07:10:27,757 - INFO - 🚨 Test Cross-Category Errors: 0.000
2025-10-29 07:10:27,757 - INFO - 🏭 Production Readiness: ⚠️  NEEDS IMPROVEMENT
2025-10-29 07:10:27,757 - INFO - 💾 ONNX export configuration saved to models/distilbert_distilled/onnx_export_config.json
2025-10-29 07:10:27,757 - INFO - 🎯 Ready for ONNX export! Run the command in onnx_export_config.json
2025-10-29 07:10:27,758 - INFO - ✅ Training pipeline complete!
2025-10-29 07:10:27,758 - INFO - 📊 Comprehensive summary saved to: models/distilbert_distilled/training_completion_summary.json
2025-10-29 07:10:27,758 - INFO - 🎯 Model ready for production deployment and ONNX export!
```

# INTENT_CATEGORIES_LIST:
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

# Better labels (where Gemini 2.5 Flash and GPT 4o mini agree):  

2025-10-29 22:28:16,220 - INFO - 
📋 VALIDATION CLASSIFICATION REPORT
2025-10-29 22:28:16,221 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.88      0.91      0.89       394
           B       0.92      0.85      0.88       157
           C       0.66      0.59      0.62        88
           D       0.90      0.91      0.91       276
           E       0.63      0.82      0.71       118
           F       0.90      0.77      0.83        56
           G       0.72      0.68      0.70        19
           H       0.78      0.33      0.47        21
           I       0.92      0.81      0.86        54
           J       0.79      0.83      0.81        88
           K       0.95      0.58      0.72        36
           L       0.33      0.33      0.33        24
           M       0.73      0.79      0.76        14

    accuracy                           0.83      1345
   macro avg       0.78      0.71      0.73      1345
weighted avg       0.83      0.83      0.83      1345

2025-10-29 22:28:17,243 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled/validation_confusion_matrix.png
2025-10-29 22:28:17,243 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-10-29 22:28:17,243 - INFO -    Total predictions: 1345
2025-10-29 22:28:17,243 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 22:28:17,244 - INFO -    Cross-category error rate: 0.00%
2025-10-29 22:28:17,244 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 22:28:17,244 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 22:28:17,244 - INFO -    Overall accuracy: 82.83%
2025-10-29 22:28:17,245 - INFO - 💾 Validation results saved to models/distilbert_distilled/validation_results.json
2025-10-29 22:28:17,245 - INFO - 🧪 Evaluating on held-out test set...
2025-10-29 22:28:17,246 - INFO - 📊 Running comprehensive evaluation on test set...


📋 TEST CLASSIFICATION REPORT
2025-10-29 22:28:40,083 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.89      0.92      0.90       405
           B       0.90      0.86      0.88       174
           C       0.72      0.60      0.66       100
           D       0.87      0.96      0.91       276
           E       0.69      0.76      0.72       117
           F       0.91      0.76      0.83        51
           G       0.75      0.60      0.67        20
           H       0.55      0.35      0.43        17
           I       0.87      0.87      0.87        46
           J       0.75      0.81      0.78        74
           K       0.96      0.74      0.84        35
           L       0.23      0.19      0.21        16
           M       0.75      0.60      0.67        15

    accuracy                           0.84      1346
   macro avg       0.76      0.69      0.72      1346
weighted avg       0.84      0.84      0.83      1346

2025-10-29 22:28:41,085 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-10-29 22:28:41,086 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-10-29 22:28:41,086 - INFO -    Total predictions: 1346
2025-10-29 22:28:41,086 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 22:28:41,086 - INFO -    Cross-category error rate: 0.00%
2025-10-29 22:28:41,086 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 22:28:41,086 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 22:28:41,086 - INFO -    Overall accuracy: 83.80%
2025-10-29 22:28:41,088 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json


# BERT Multilingual: 

📋 VALIDATION CLASSIFICATION REPORT
2025-10-29 23:20:28,699 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.69      0.82      0.75       394
           B       0.81      0.80      0.81       157
           C       0.31      0.12      0.18        88
           D       0.85      0.85      0.85       276
           E       0.43      0.64      0.51       118
           F       0.85      0.59      0.69        56
           G       0.50      0.32      0.39        19
           H       1.00      0.05      0.09        21
           I       0.82      0.76      0.79        54
           J       0.73      0.69      0.71        88
           K       0.59      0.28      0.38        36
           L       0.30      0.25      0.27        24
           M       0.53      0.57      0.55        14

    accuracy                           0.70      1345
   macro avg       0.65      0.52      0.54      1345
weighted avg       0.70      0.70      0.68      1345

2025-10-29 23:20:29,713 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled/validation_confusion_matrix.png
2025-10-29 23:20:29,713 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-10-29 23:20:29,714 - INFO -    Total predictions: 1345
2025-10-29 23:20:29,714 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 23:20:29,714 - INFO -    Cross-category error rate: 0.00%
2025-10-29 23:20:29,714 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 23:20:29,714 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 23:20:29,714 - INFO -    Overall accuracy: 69.52%
2025-10-29 23:20:29,715 - INFO - 💾 Validation results saved to models/distilbert_distilled/validation_results.json
2025-10-29 23:20:29,715 - INFO - 🧪 Evaluating on held-out test set...
2025-10-29 23:20:29,716 - INFO - 📊 Running comprehensive evaluation on test set...


📋 TEST CLASSIFICATION REPORT
2025-10-29 23:20:53,345 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.72      0.80      0.76       405
           B       0.83      0.79      0.81       174
           C       0.35      0.14      0.20       100
           D       0.82      0.90      0.86       276
           E       0.40      0.60      0.48       117
           F       0.84      0.53      0.65        51
           G       0.46      0.30      0.36        20
           H       0.00      0.00      0.00        17
           I       0.78      0.67      0.72        46
           J       0.73      0.72      0.72        74
           K       0.62      0.43      0.51        35
           L       0.15      0.19      0.17        16
           M       0.75      0.40      0.52        15

    accuracy                           0.69      1346
   macro avg       0.57      0.50      0.52      1346
weighted avg       0.68      0.69      0.68      1346

2025-10-29 23:20:54,339 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-10-29 23:20:54,340 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-10-29 23:20:54,340 - INFO -    Total predictions: 1346
2025-10-29 23:20:54,340 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 23:20:54,341 - INFO -    Cross-category error rate: 0.00%
2025-10-29 23:20:54,341 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 23:20:54,341 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 23:20:54,341 - INFO -    Overall accuracy: 69.39%
2025-10-29 23:20:54,342 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json
2025-10-29 23:20:54,342 - INFO - 

# DistilBERT Multilingual: 
📋 VALIDATION CLASSIFICATION REPORT
2025-10-29 23:48:17,895 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.88      0.91      0.90       394
           B       0.89      0.82      0.85       157
           C       0.62      0.66      0.64        88
           D       0.86      0.93      0.90       276
           E       0.65      0.75      0.70       118
           F       0.86      0.77      0.81        56
           G       0.71      0.63      0.67        19
           H       0.50      0.33      0.40        21
           I       0.91      0.78      0.84        54
           J       0.87      0.74      0.80        88
           K       0.79      0.53      0.63        36
           L       0.27      0.29      0.28        24
           M       0.79      0.79      0.79        14

    accuracy                           0.82      1345
   macro avg       0.74      0.69      0.71      1345
weighted avg       0.82      0.82      0.81      1345

2025-10-29 23:48:18,921 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled/validation_confusion_matrix.png
2025-10-29 23:48:18,922 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-10-29 23:48:18,922 - INFO -    Total predictions: 1345
2025-10-29 23:48:18,922 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 23:48:18,922 - INFO -    Cross-category error rate: 0.00%
2025-10-29 23:48:18,923 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 23:48:18,923 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 23:48:18,923 - INFO -    Overall accuracy: 81.56%
2025-10-29 23:48:18,924 - INFO - 💾 Validation results saved to models/distilbert_distilled/validation_results.json
2025-10-29 23:48:18,924 - INFO - 🧪 Evaluating on held-out test set...
2025-10-29 23:48:18,925 - INFO - 📊 Running comprehensive evaluation on test set...


📋 TEST CLASSIFICATION REPORT
2025-10-29 23:48:38,607 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.87      0.88      0.88       405
           B       0.90      0.84      0.87       174
           C       0.57      0.60      0.58       100
           D       0.89      0.93      0.91       276
           E       0.62      0.68      0.65       117
           F       0.94      0.67      0.78        51
           G       0.68      0.75      0.71        20
           H       0.38      0.47      0.42        17
           I       0.92      0.78      0.85        46
           J       0.84      0.76      0.79        74
           K       0.97      0.80      0.88        35
           L       0.22      0.38      0.28        16
           M       0.89      0.53      0.67        15

    accuracy                           0.81      1346
   macro avg       0.75      0.70      0.71      1346
weighted avg       0.82      0.81      0.81      1346

2025-10-29 23:48:39,610 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-10-29 23:48:39,611 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-10-29 23:48:39,611 - INFO -    Total predictions: 1346
2025-10-29 23:48:39,612 - INFO -    Cross-category errors (NSFW/SFW): 0
2025-10-29 23:48:39,613 - INFO -    Cross-category error rate: 0.00%
2025-10-29 23:48:39,613 - INFO -    NSFW recall errors (missed NSFW): 0
2025-10-29 23:48:39,613 - INFO -    NSFW precision errors (false NSFW): 0
2025-10-29 23:48:39,613 - INFO -    Overall accuracy: 81.05%
2025-10-29 23:48:39,615 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json
2025-10-29 23:48:39,615 - INFO - 
============================================================
2025-10-29 23:48:39,615 - INFO - TRAINING COMPLETION SUMMARY
2025-10-29 23:48:39,615 - INFO - ============================================================
2025-10-29 23:48:39,615 - INFO - 🎯 Validation Accuracy: 0.816
2025-10-29 23:48:39,616 - INFO - 🧪 Test Accuracy: 0.811
2025-10-29 23:48:39,616 - INFO - 📉 Accuracy Drop: 0.005 (✅ OK)
2025-10-29 23:48:39,616 - INFO - 🚨 Validation Cross-Category Errors: 0.000
2025-10-29 23:48:39,616 - INFO - 🚨 Test Cross-Category Errors: 0.000
2025-10-29 23:48:39,616 - INFO - 🏭 Production Readiness: ✅ READY
2025-10-29 23:48:39,616 - INFO - 🚀 Exporting model to ONNX format...


# DistilBERT 1024 Tokens (simpler message parsing, uncased base): 

📋 VALIDATION CLASSIFICATION REPORT
2025-10-30 23:58:42,609 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.87      0.87      0.87       410
           B       0.82      0.94      0.87       165
           C       0.76      0.56      0.65       126
           D       0.92      0.94      0.93       285
           E       0.65      0.75      0.70       149
           F       0.85      0.90      0.87        58
           G       0.93      0.75      0.83        36
           H       0.67      0.30      0.41        20
           I       0.87      0.83      0.85        64
           J       0.80      0.77      0.78       104
           K       0.78      0.67      0.72        27
           L       0.21      0.25      0.23        20
           M       0.69      0.65      0.67        17

    accuracy                           0.82      1481
   macro avg       0.76      0.71      0.72      1481
weighted avg       0.82      0.82      0.82      1481

2025-10-30 23:58:43,661 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled_1024/validation_confusion_matrix.png
2025-10-30 23:58:43,662 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-10-30 23:58:43,662 - INFO -    Total predictions: 1481
2025-10-30 23:58:43,662 - INFO -    Cross-category errors (NSFW/SFW): 63
2025-10-30 23:58:43,663 - INFO -    Cross-category error rate: 4.25%
2025-10-30 23:58:43,663 - INFO -    NSFW recall errors (missed NSFW): 30
2025-10-30 23:58:43,663 - INFO -    NSFW precision errors (false NSFW): 33
2025-10-30 23:58:43,663 - INFO -    Overall accuracy: 82.11%
2025-10-30 23:58:43,664 - INFO - 💾 Validation results saved to models/distilbert_distilled_1024/validation_results.json


TEST CLASSIFICATION REPORT
2025-10-30 23:59:24,439 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.88      0.85      0.87       431
           B       0.73      0.90      0.81       172
           C       0.62      0.50      0.55       121
           D       0.93      0.92      0.92       294
           E       0.63      0.78      0.70       128
           F       0.79      0.80      0.80        46
           G       0.81      0.74      0.78        35
           H       0.87      0.54      0.67        24
           I       0.97      0.95      0.96        62
           J       0.88      0.79      0.83        94
           K       0.97      0.73      0.83        44
           L       0.42      0.53      0.47        19
           M       0.54      0.54      0.54        13

    accuracy                           0.82      1483
   macro avg       0.77      0.74      0.75      1483
weighted avg       0.82      0.82      0.82      1483

2025-10-30 23:59:25,457 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled_1024/test_confusion_matrix.png
2025-10-30 23:59:25,457 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-10-30 23:59:25,458 - INFO -    Total predictions: 1483
2025-10-30 23:59:25,458 - INFO -    Cross-category errors (NSFW/SFW): 66
2025-10-30 23:59:25,459 - INFO -    Cross-category error rate: 4.45%
2025-10-30 23:59:25,460 - INFO -    NSFW recall errors (missed NSFW): 40
2025-10-30 23:59:25,460 - INFO -    NSFW precision errors (false NSFW): 26
2025-10-30 23:59:25,460 - INFO -    Overall accuracy: 81.66%

# BERT 1024 Tokens (simpler message parsing, uncased base): 

 VALIDATION CLASSIFICATION REPORT
2025-10-31 04:50:35,977 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.90      0.90      0.90       410
           B       0.89      0.91      0.90       165
           C       0.84      0.64      0.73       126
           D       0.92      0.92      0.92       285
           E       0.62      0.85      0.72       149
           F       0.94      0.88      0.91        58
           G       0.97      0.81      0.88        36
           H       0.82      0.45      0.58        20
           I       0.93      0.81      0.87        64
           J       0.84      0.74      0.79       104
           K       0.86      0.67      0.75        27
           L       0.27      0.45      0.34        20
           M       0.68      0.88      0.77        17

    accuracy                           0.84      1481
   macro avg       0.81      0.76      0.77      1481
weighted avg       0.86      0.84      0.85      1481

2025-10-31 04:50:36,998 - INFO - 💾 Validation confusion matrix saved to models/bert_distilled_1024/validation_confusion_matrix.png
2025-10-31 04:50:36,999 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-10-31 04:50:36,999 - INFO -    Total predictions: 1481
2025-10-31 04:50:36,999 - INFO -    Cross-category errors (NSFW/SFW): 62
2025-10-31 04:50:36,999 - INFO -    Cross-category error rate: 4.19%
2025-10-31 04:50:37,000 - INFO -    NSFW recall errors (missed NSFW): 35
2025-10-31 04:50:37,000 - INFO -    NSFW precision errors (false NSFW): 27
2025-10-31 04:50:37,000 - INFO -    Overall accuracy: 84.40%
2025-10-31 04:50:37,001 - INFO - 💾 Validation results saved to models/bert_distilled_1024/validation_results.json
2025-10-31 04:50:37,001 - INFO - 🧪 Evaluating on held-out test set...
2025-10-31 04:50:37,001 - INFO - 📊 Running comprehensive evaluation on test set...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 93/93 [00:33<00:00,  2.75it/s]
2025-10-31 04:51:27,188 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-10-31 04:51:27,189 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.89      0.91      0.90       431
           B       0.88      0.87      0.87       172
           C       0.75      0.68      0.71       121
           D       0.95      0.93      0.94       294
           E       0.59      0.81      0.68       128
           F       0.90      0.83      0.86        46
           G       0.85      0.63      0.72        35
           H       0.80      0.67      0.73        24
           I       0.97      0.92      0.94        62
           J       0.92      0.82      0.87        94
           K       0.94      0.73      0.82        44
           L       0.33      0.47      0.39        19
           M       0.64      0.54      0.58        13

    accuracy                           0.85      1483
   macro avg       0.80      0.75      0.77      1483
weighted avg       0.86      0.85      0.85      1483

2025-10-31 04:51:28,202 - INFO - 💾 Test confusion matrix saved to models/bert_distilled_1024/test_confusion_matrix.png
2025-10-31 04:51:28,202 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-10-31 04:51:28,203 - INFO -    Total predictions: 1483
2025-10-31 04:51:28,203 - INFO -    Cross-category errors (NSFW/SFW): 55
2025-10-31 04:51:28,203 - INFO -    Cross-category error rate: 3.71%
2025-10-31 04:51:28,203 - INFO -    NSFW recall errors (missed NSFW): 38
2025-10-31 04:51:28,203 - INFO -    NSFW precision errors (false NSFW): 17
2025-10-31 04:51:28,203 - INFO -    Overall accuracy: 84.76%
2025-10-31 04:51:28,204 - INFO - 💾 Test results saved to models/bert_distilled_1024/test_results.json
2025-10-31 04:51:28,205 - INFO - 