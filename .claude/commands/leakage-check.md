---
description: Audit the ML code for data leakage
---

Use the reviewer agent to audit app/ml/ and app/core/ for:
- target leakage in features
- preprocessing fitted before the train/test split or outside a Pipeline
- test data touching tuning or model selection
- CV folds that are not stratified for classification
List each issue with file, line and a fix.
