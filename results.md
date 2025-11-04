# 1024 Token Distil BERT with Student Teacher Learning (11/02/25)
2025-11-02 19:47:12,221 - INFO - 
📋 VALIDATION CLASSIFICATION REPORT
2025-11-02 19:47:12,221 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.85      0.96      0.90       658
           B       0.88      0.89      0.88       281
           C       0.73      0.66      0.69       191
           D       0.93      0.90      0.92       417
           E       0.78      0.72      0.75       156
           F       0.95      0.85      0.90        81
           G       0.86      0.63      0.73        51
           H       0.74      0.52      0.61        27
           I       0.85      0.81      0.83        93
           J       0.89      0.75      0.82       141
           K       0.88      0.75      0.81        57
           L       0.39      0.45      0.42        40
           M       0.68      0.79      0.73        29

    accuracy                           0.85      2222
   macro avg       0.80      0.75      0.77      2222
weighted avg       0.85      0.85      0.84      2222

2025-11-02 19:47:13,237 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled_1024/validation_confusion_matrix.png
2025-11-02 19:47:13,237 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-11-02 19:47:13,238 - INFO -    Total predictions: 2222
2025-11-02 19:47:13,238 - INFO -    Cross-category errors (NSFW/SFW): 80
2025-11-02 19:47:13,238 - INFO -    Cross-category error rate: 3.60%
2025-11-02 19:47:13,238 - INFO -    NSFW recall errors (missed NSFW): 54
2025-11-02 19:47:13,238 - INFO -    NSFW precision errors (false NSFW): 26
2025-11-02 19:47:13,238 - INFO -    Overall accuracy: 84.65%
2025-11-02 19:47:13,239 - INFO - 💾 Validation results saved to models/distilbert_distilled_1024/validation_results.json
2025-11-02 19:47:13,239 - INFO - 🧪 Evaluating on held-out test set...
2025-11-02 19:47:13,240 - INFO - 📊 Running comprehensive evaluation on test set...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 278/278 [00:30<00:00,  9.23it/s]
2025-11-02 19:47:59,132 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-11-02 19:47:59,132 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.84      0.96      0.90       689
           B       0.84      0.84      0.84       272
           C       0.68      0.58      0.63       202
           D       0.92      0.88      0.90       418
           E       0.74      0.71      0.72       194
           F       0.94      0.84      0.89        70
           G       0.86      0.79      0.83        48
           H       0.76      0.48      0.59        33
           I       0.88      0.82      0.85        80
           J       0.87      0.71      0.78       119
           K       0.89      0.85      0.87        46
           L       0.26      0.42      0.32        26
           M       0.56      0.70      0.62        27

    accuracy                           0.83      2224
   macro avg       0.77      0.74      0.75      2224
weighted avg       0.83      0.83      0.83      2224

2025-11-02 19:48:00,135 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled_1024/test_confusion_matrix.png
2025-11-02 19:48:00,135 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-11-02 19:48:00,136 - INFO -    Total predictions: 2224
2025-11-02 19:48:00,136 - INFO -    Cross-category errors (NSFW/SFW): 110
2025-11-02 19:48:00,136 - INFO -    Cross-category error rate: 4.95%
2025-11-02 19:48:00,136 - INFO -    NSFW recall errors (missed NSFW): 73
2025-11-02 19:48:00,137 - INFO -    NSFW precision errors (false NSFW): 37
2025-11-02 19:48:00,137 - INFO -    Overall accuracy: 82.73%
2025-11-02 19:48:00,138 - INFO - 💾 Test results saved to models/distilbert_distilled_1024/test_results.json
2025-11-02 19:48:00,138 - INFO - 
============================================================
2025-11-02 19:48:00,138 - INFO - TRAINING COMPLETION SUMMARY
2025-11-02 19:48:00,138 - INFO - ============================================================
2025-11-02 19:48:00,138 - INFO - 🎯 Validation Accuracy: 0.847
2025-11-02 19:48:00,139 - INFO - 🧪 Test Accuracy: 0.827
2025-11-02 19:48:00,139 - INFO - 📉 Accuracy Drop: 0.019 (✅ OK)
2025-11-02 19:48:00,139 - INFO - 🚨 Validation Cross-Category Errors: 0.036
2025-11-02 19:48:00,139 - INFO - 🚨 Test Cross-Category Errors: 0.049
2025-11-02 19:48:00,139 - INFO - 🏭 Production Readiness: ✅ READY
2025-11-02 19:48:00,140 - INFO - 🚀 Exporting model to ONNX format...
2025-11-02 19:48:00,212 - INFO - 🔧 Re-extending position embeddings for ONNX export...

# 1024 Token BERT with Student Teacher Learning (11/02/25):

📋 VALIDATION CLASSIFICATION REPORT
2025-11-02 19:47:12,221 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.85      0.96      0.90       658
           B       0.88      0.89      0.88       281
           C       0.73      0.66      0.69       191
           D       0.93      0.90      0.92       417
           E       0.78      0.72      0.75       156
           F       0.95      0.85      0.90        81
           G       0.86      0.63      0.73        51
           H       0.74      0.52      0.61        27
           I       0.85      0.81      0.83        93
           J       0.89      0.75      0.82       141
           K       0.88      0.75      0.81        57
           L       0.39      0.45      0.42        40
           M       0.68      0.79      0.73        29

    accuracy                           0.85      2222
   macro avg       0.80      0.75      0.77      2222
weighted avg       0.85      0.85      0.84      2222

2025-11-02 19:47:13,237 - INFO - 💾 Validation confusion matrix saved to models/distilbert_distilled_1024/validation_confusion_matrix.png
2025-11-02 19:47:13,237 - INFO - 
🚨 VALIDATION BUSINESS IMPACT ANALYSIS:
2025-11-02 19:47:13,238 - INFO -    Total predictions: 2222
2025-11-02 19:47:13,238 - INFO -    Cross-category errors (NSFW/SFW): 80
2025-11-02 19:47:13,238 - INFO -    Cross-category error rate: 3.60%
2025-11-02 19:47:13,238 - INFO -    NSFW recall errors (missed NSFW): 54
2025-11-02 19:47:13,238 - INFO -    NSFW precision errors (false NSFW): 26
2025-11-02 19:47:13,238 - INFO -    Overall accuracy: 84.65%
2025-11-02 19:47:13,239 - INFO - 💾 Validation results saved to models/distilbert_distilled_1024/validation_results.json
2025-11-02 19:47:13,239 - INFO - 🧪 Evaluating on held-out test set...
2025-11-02 19:47:13,240 - INFO - 📊 Running comprehensive evaluation on test set...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 278/278 [00:30<00:00,  9.23it/s]
2025-11-02 19:47:59,132 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-11-02 19:47:59,132 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.84      0.96      0.90       689
           B       0.84      0.84      0.84       272
           C       0.68      0.58      0.63       202
           D       0.92      0.88      0.90       418
           E       0.74      0.71      0.72       194
           F       0.94      0.84      0.89        70
           G       0.86      0.79      0.83        48
           H       0.76      0.48      0.59        33
           I       0.88      0.82      0.85        80
           J       0.87      0.71      0.78       119
           K       0.89      0.85      0.87        46
           L       0.26      0.42      0.32        26
           M       0.56      0.70      0.62        27

    accuracy                           0.83      2224
   macro avg       0.77      0.74      0.75      2224
weighted avg       0.83      0.83      0.83      2224

2025-11-02 19:48:00,135 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled_1024/test_confusion_matrix.png
2025-11-02 19:48:00,135 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-11-02 19:48:00,136 - INFO -    Total predictions: 2224
2025-11-02 19:48:00,136 - INFO -    Cross-category errors (NSFW/SFW): 110
2025-11-02 19:48:00,136 - INFO -    Cross-category error rate: 4.95%
2025-11-02 19:48:00,136 - INFO -    NSFW recall errors (missed NSFW): 73
2025-11-02 19:48:00,137 - INFO -    NSFW precision errors (false NSFW): 37
2025-11-02 19:48:00,137 - INFO -    Overall accuracy: 82.73%
2025-11-02 19:48:00,138 - INFO - 💾 Test results saved to models/distilbert_distilled_1024/test_results.json
2025-11-02 19:48:00,138 - INFO - 
============================================================
2025-11-02 19:48:00,138 - INFO - TRAINING COMPLETION SUMMARY
2025-11-02 19:48:00,138 - INFO - ============================================================
2025-11-02 19:48:00,138 - INFO - 🎯 Validation Accuracy: 0.847
2025-11-02 19:48:00,139 - INFO - 🧪 Test Accuracy: 0.827
2025-11-02 19:48:00,139 - INFO - 📉 Accuracy Drop: 0.019 (✅ OK)
2025-11-02 19:48:00,139 - INFO - 🚨 Validation Cross-Category Errors: 0.036
2025-11-02 19:48:00,139 - INFO - 🚨 Test Cross-Category Errors: 0.049
2025-11-02 19:48:00,139 - INFO - 🏭 Production Readiness: ✅ READY
2025-11-02 19:48:00,140 - INFO - 🚀 Exporting model to ONNX format...
2025-11-02 19:48:00,212 - INFO - 🔧 Re-extending position embeddings for ONNX export...
2025-11-02 19:48:00,212 - WARNING - new_max_length 1024 <= existing 1024, no extension needed