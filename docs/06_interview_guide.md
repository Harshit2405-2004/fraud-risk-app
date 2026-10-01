# 06 · Interview Guide — Pitch, Demo, and Q&A

## The 30-second pitch (memorise the shape, not the words)

> "I built an end-to-end fraud detection system. I trained and compared five models on a million card
> transactions where only 8.7 % are fraud, handled the imbalance with SMOTE on the training set only,
> and ended with a stacked ensemble that catches about 99.9 % of fraud on a held-out test set at about
> 97 % precision. Then I deployed it as a Streamlit app — you can try it yourself at *[your URL]*.
> I'll also tell you where the results are too good to be true and what I'd change for production."

## The 3-minute live demo script

1. **(20 s) Frame it.** "Enter a transaction, the model returns a fraud risk score."
2. **(40 s) Normal case.** Pick *Everyday chip & PIN purchase* → low score. "Chip, PIN, near home, normal amount."
3. **(40 s) Suspicious case.** Pick *far away, online, 7x spend* → flagged. "Unusual amount, far from home, no chip or PIN."
4. **(40 s) Threshold slider.** "A bank decides how much risk to tolerate. Lower this and we catch more
   fraud but flag more honest customers — that's the precision/recall trade-off."
5. **(30 s) What-if chart.** "Holding everything else fixed, here's how the score responds as the
   purchase amount grows."
6. **(10 s) Batch tab.** "It also scores CSV files." Stop there unless asked.

Finish with the **Limitations** box. Volunteering weaknesses reads as maturity.

---

## Likely questions and strong answers

**Q1. Why is accuracy a bad metric here?**
Only 8.7 % of rows are fraud, so predicting "legitimate" always gives ~91 % accuracy. I used fraud-class
precision, recall and F1 instead.

**Q2. What is SMOTE and why only on the training set?**
It creates synthetic minority examples by interpolating between neighbouring fraud cases. If applied
before the split, synthetic copies of test points could leak into training and inflate scores. The test
set must reflect reality.

**Q3. Why did you scale only three columns?**
The three numeric columns have very different ranges; scale-sensitive models (SVM, logistic regression)
need them standardised. The four 0/1 columns are already on a comparable scale. Tree models don't need
scaling at all.

**Q4. Why cap the outliers instead of deleting them?**
Deleting rows could remove real fraud cases (extreme values are often fraud-related). Capping keeps every
row while limiting the influence of extreme values. I used about the 98th/99th percentiles.

**Q5. Your scores are ~99.8 %. Isn't that suspicious?**
Yes, and I'd say so first. The dataset appears synthetic and highly separable. Real fraud data has
overlapping classes, concept drift and a far lower fraud rate. I treat this as a demonstration of the
pipeline, not a claim about real-world performance.

**Q6. Why a stacked model if Random Forest is almost as good?**
It scored marginally higher (F1 0.9864 vs 0.9863), but that gap is negligible. For deployment I'd pick
the simpler, smaller, faster model. *(Adjust to the model you actually deployed.)*

**Q7. Why did Logistic Regression do so badly?**
Fraud depends on non-linear combinations of features (e.g. high amount *and* online *and* no PIN). A
linear model can't represent that; trees and kernels can.

**Q8. How would you choose the threshold in production?**
From costs: cost of a missed fraud vs. cost of a false alarm (customer friction, review staff). I'd
pick the threshold that minimises expected cost on a validation set, or use a precision target.
The app's slider lets stakeholders explore this.

**Q9. Are the output probabilities reliable?**
No — SMOTE changed the class balance, so they're risk scores. I'd calibrate (Platt or isotonic) on
untouched validation data before presenting them as probabilities.

**Q10. How do you make sure training and serving preprocessing match?**
Same capping thresholds in code, and the *fitted* scaler is saved and reused. I also checked that the app
reproduces the notebook's predictions on the same rows.

**Q11. What would break in production?**
Concept drift (fraudsters adapt), data pipeline changes (a feature's meaning/unit changes), and
library-version mismatch when loading the model. I pinned versions in `requirements.txt`. I'd add
monitoring of input distributions and fraud rate.

**Q12. How does the deployment work?**
Code and model files live in a GitHub repo; Streamlit Community Cloud clones it, installs
`requirements.txt`, and runs `app/app.py`. The model is loaded once with `st.cache_resource`.

**Q13. Is this safe for real card data?**
Not as built — it's a demo. A real system needs authentication, encryption, audit logging, and
compliance with standards like PCI-DSS. Also, loading pickled models is only safe from trusted sources.

**Q14. What did you do about overfitting?**
Evaluation on a held-out 20 % that was never resampled. I didn't do cross-validation or tuning —
that's a limitation and the first thing I'd add.

**Q15. What features mattered most?**
*(Run the feature-importance snippet from Guide 02 and answer with your real numbers.)*

**Q16. What would you do with another week?**
Time-based split + cross-validation, calibration, PR-AUC reporting, SHAP explanations in the app,
a FastAPI service for the model, and drift monitoring.

---

## "Tell me about a challenge" — a ready story

*"The full stacked model was too big to host for free — GitHub caps files at 100 MB. I compared a
smaller Random Forest and found it matched the F1 score to within a tiny margin, so I shipped the
smaller model. It taught me that the best model on paper isn't always the best model to deploy."*

Only use this if it's true for you (check your own file sizes and scores).

## Be ready to open the code

They may ask to see `preprocess.py`. Be able to explain, in order: clipping → scaling with the saved
scaler → appending binary columns, and why the column order matters.

## Day-of checklist

- [ ] Open the live URL 10 minutes early (wakes the app)
- [ ] Backup: local copy running + screen recording
- [ ] Notebook open in another tab (for EDA/metrics questions)
- [ ] You can state the five numbers: 1M rows · 8.7 % fraud · 800k/200k split · stacked fraud
      recall 0.999 · precision 0.974
