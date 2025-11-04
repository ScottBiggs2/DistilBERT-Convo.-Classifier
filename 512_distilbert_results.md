# 512 Tokens with $\alpha = 0.3$: 

2025-11-03 22:54:09,495 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-11-03 22:54:09,495 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.87      0.95      0.91       689
           B       0.84      0.87      0.86       272
           C       0.75      0.61      0.67       202
           D       0.91      0.89      0.90       418
           E       0.72      0.81      0.76       194
           F       0.90      0.89      0.89        70
           G       0.75      0.75      0.75        48
           H       0.62      0.30      0.41        33
           I       0.88      0.85      0.87        80
           J       0.85      0.74      0.79       119
           K       0.77      0.72      0.74        46
           L       0.40      0.31      0.35        26
           M       0.67      0.52      0.58        27

    accuracy                           0.84      2224
   macro avg       0.76      0.71      0.73      2224
weighted avg       0.83      0.84      0.83      2224

2025-11-03 22:54:10,507 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-11-03 22:54:10,508 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-11-03 22:54:10,508 - INFO -    Total predictions: 2224
2025-11-03 22:54:10,508 - INFO -    Cross-category errors (NSFW/SFW): 106
2025-11-03 22:54:10,508 - INFO -    Cross-category error rate: 4.77%
2025-11-03 22:54:10,508 - INFO -    NSFW recall errors (missed NSFW): 70
2025-11-03 22:54:10,508 - INFO -    NSFW precision errors (false NSFW): 36
2025-11-03 22:54:10,509 - INFO -    Overall accuracy: 83.77%
2025-11-03 22:54:10,510 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json
2025-11-03 22:54:10,510 - INFO - 
============================================================
2025-11-03 22:54:10,510 - INFO - TRAINING COMPLETION SUMMARY
2025-11-03 22:54:10,510 - INFO - ============================================================
2025-11-03 22:54:10,511 - INFO - 🎯 Validation Accuracy: 0.857
2025-11-03 22:54:10,511 - INFO - 🧪 Test Accuracy: 0.838
2025-11-03 22:54:10,511 - INFO - 📉 Accuracy Drop: 0.020 (✅ OK)
2025-11-03 22:54:10,511 - INFO - 🚨 Validation Cross-Category Errors: 0.036
2025-11-03 22:54:10,511 - INFO - 🚨 Test Cross-Category Errors: 0.048
2025-11-03 22:54:10,511 - INFO - 🏭 Production Readiness: ✅ READY
2025-11-03 22:54:10,512 - INFO - 🚀 Exporting model to ONNX format...
W1103 22:54:10.756000 5979 torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 11 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
W1103 22:54:11.404000 5979 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::nms
[torch.onnx] Obtain model graph for `DistilBertForSequenceClassification([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `DistilBertForSequenceClassification([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...

# 512 Tokens with $\alpha = 0.7$:

2025-11-03 23:29:01,183 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-11-03 23:29:01,184 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.85      0.93      0.89       689
           B       0.85      0.91      0.88       272
           C       0.72      0.64      0.68       202
           D       0.91      0.88      0.90       418
           E       0.70      0.71      0.70       194
           F       0.97      0.83      0.89        70
           G       0.73      0.69      0.71        48
           H       0.68      0.45      0.55        33
           I       0.92      0.84      0.88        80
           J       0.86      0.80      0.83       119
           K       0.86      0.70      0.77        46
           L       0.28      0.31      0.29        26
           M       0.67      0.52      0.58        27

    accuracy                           0.83      2224
   macro avg       0.77      0.71      0.73      2224
weighted avg       0.83      0.83      0.83      2224

2025-11-03 23:29:02,147 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-11-03 23:29:02,148 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-11-03 23:29:02,148 - INFO -    Total predictions: 2224
2025-11-03 23:29:02,148 - INFO -    Cross-category errors (NSFW/SFW): 114
2025-11-03 23:29:02,148 - INFO -    Cross-category error rate: 5.13%
2025-11-03 23:29:02,148 - INFO -    NSFW recall errors (missed NSFW): 71
2025-11-03 23:29:02,149 - INFO -    NSFW precision errors (false NSFW): 43
2025-11-03 23:29:02,149 - INFO -    Overall accuracy: 82.96%
2025-11-03 23:29:02,150 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json
2025-11-03 23:29:02,150 - INFO - 
============================================================
2025-11-03 23:29:02,150 - INFO - TRAINING COMPLETION SUMMARY
2025-11-03 23:29:02,150 - INFO - ============================================================
2025-11-03 23:29:02,150 - INFO - 🎯 Validation Accuracy: 0.846
2025-11-03 23:29:02,151 - INFO - 🧪 Test Accuracy: 0.830
2025-11-03 23:29:02,151 - INFO - 📉 Accuracy Drop: 0.016 (✅ OK)
2025-11-03 23:29:02,151 - INFO - 🚨 Validation Cross-Category Errors: 0.038
2025-11-03 23:29:02,151 - INFO - 🚨 Test Cross-Category Errors: 0.051
2025-11-03 23:29:02,151 - INFO - 🏭 Production Readiness: ⚠️  NEEDS IMPROVEMENT
2025-11-03 23:29:02,152 - INFO - 🚀 Exporting model to ONNX format...

# 512 Tokens with $\alpha = 0.0$:

2025-11-03 23:57:10,775 - INFO - 
📋 TEST CLASSIFICATION REPORT
2025-11-03 23:57:10,775 - INFO - ==================================================
              precision    recall  f1-score   support

           A       0.89      0.92      0.91       689
           B       0.86      0.86      0.86       272
           C       0.69      0.70      0.69       202
           D       0.92      0.87      0.90       418
           E       0.71      0.81      0.76       194
           F       0.95      0.80      0.87        70
           G       0.72      0.75      0.73        48
           H       0.75      0.45      0.57        33
           I       0.89      0.90      0.89        80
           J       0.75      0.81      0.78       119
           K       0.80      0.80      0.80        46
           L       0.39      0.27      0.32        26
           M       0.67      0.52      0.58        27

    accuracy                           0.84      2224
   macro avg       0.77      0.73      0.74      2224
weighted avg       0.84      0.84      0.84      2224

2025-11-03 23:57:11,742 - INFO - 💾 Test confusion matrix saved to models/distilbert_distilled/test_confusion_matrix.png
2025-11-03 23:57:11,743 - INFO - 
🚨 TEST BUSINESS IMPACT ANALYSIS:
2025-11-03 23:57:11,744 - INFO -    Total predictions: 2224
2025-11-03 23:57:11,744 - INFO -    Cross-category errors (NSFW/SFW): 115
2025-11-03 23:57:11,744 - INFO -    Cross-category error rate: 5.17%
2025-11-03 23:57:11,744 - INFO -    NSFW recall errors (missed NSFW): 67
2025-11-03 23:57:11,744 - INFO -    NSFW precision errors (false NSFW): 48
2025-11-03 23:57:11,744 - INFO -    Overall accuracy: 83.77%
2025-11-03 23:57:11,745 - INFO - 💾 Test results saved to models/distilbert_distilled/test_results.json
2025-11-03 23:57:11,746 - INFO - 
============================================================
2025-11-03 23:57:11,746 - INFO - TRAINING COMPLETION SUMMARY
2025-11-03 23:57:11,746 - INFO - ============================================================
2025-11-03 23:57:11,746 - INFO - 🎯 Validation Accuracy: 0.855
2025-11-03 23:57:11,746 - INFO - 🧪 Test Accuracy: 0.838
2025-11-03 23:57:11,747 - INFO - 📉 Accuracy Drop: 0.017 (✅ OK)
2025-11-03 23:57:11,747 - INFO - 🚨 Validation Cross-Category Errors: 0.038
2025-11-03 23:57:11,747 - INFO - 🚨 Test Cross-Category Errors: 0.052
2025-11-03 23:57:11,747 - INFO - 🏭 Production Readiness: ⚠️  NEEDS IMPROVEMENT
2025-11-03 23:57:11,747 - INFO - 🚀 Exporting model to ONNX format...

