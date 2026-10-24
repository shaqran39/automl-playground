---
description: Run the full pipeline on every sample CSV and report failures
---

Run every stage that exists so far (profile, clean, detect task, train) on every CSV in sample_data/.
Use a tiny tuning time budget. Give me a table: dataset, stage reached, task detected, suggested target, best model, score, errors.
Then suggest fixes for any failures.
