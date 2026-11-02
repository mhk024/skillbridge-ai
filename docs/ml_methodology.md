# ML Methodology (work in progress)

Status: Phase 5A (data + features) implemented. Model training and evaluation: **pending (Phase 5B)**.
No metrics are reported in this document until they come from an executed training run.

## 1. Prediction task
Estimate the probability that a student's next attempt on a question will be **correct**
(passes all tests). One row = one attempt. Target: `is_correct` in {0, 1}.

## 2. Dataset
- **Current development dataset: SYNTHETIC** (`src/ml/generate_synthetic.py`, name
  `skillbridge_synthetic_v1`). Generated from a simulation with documented assumptions
  (see `data/ml/synthetic_attempts.meta.json` after generation). It is **not** real student data.
- **Why synthetic:** the app's own question bank is original and new users have no history, so
  there is no real data for this exact setting yet. Real attempts collected by the app
  (`submissions` table) can replace it later.
- **Planned comparison: IBM Project CodeNet.** Its metadata contains per-submission records with
  user, problem, timestamp and status, which is the right shape for a similar task. Its schema,
  license and the availability of difficulty/topic fields must be inspected before use. Pending.
- **Limitation:** scores on synthetic data show how well a model recovers the simulation's rules,
  not real-world predictive performance.

## 3. Raw table schema
`attempt_id, user_id, question_id, topic, difficulty, attempted_at, is_correct`
(the synthetic file also has `sim_true_prob`, a diagnostic never used as a feature).

## 4. Features (all from the same student's strictly earlier attempts)
| Feature | Meaning |
|---|---|
| user_prior_attempts / user_prior_accuracy | Overall history, accuracy smoothed toward 0.5 |
| recent_accuracy_5 | Accuracy over the previous 5 attempts (smoothed) |
| topic_prior_attempts / topic_prior_accuracy | History in this topic (smoothed toward the student's overall accuracy) |
| topic_prior_solved | Distinct questions already solved in this topic |
| difficulty_prior_attempts / difficulty_prior_accuracy | History at this difficulty |
| difficulty_level | easy=1, medium=2, hard=3 |
| prior_attempts_on_question | Earlier attempts on this very question (retry signal) |
| topic | Categorical, one-hot encoded with fixed categories |

Smoothing uses fixed constants (prior rate 0.5, strength 2), so nothing is learned from other users.

## 5. Leakage prevention
1. Features for attempt *i* use only that student's attempts before *i* (ordered by time, then id).
2. An attempt's own outcome is never an input to its own features (tested).
3. Later attempts never change earlier rows' features (tested).
4. Live predictions use the same code as training via a placeholder attempt (tested for parity).
5. Evaluation will split by **student** (group-aware) so one student's attempts are never in both
   train and test sets. (Phase 5B)

## 6. Planned evaluation (Phase 5B)
Logistic Regression (baseline) vs Random Forest; accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, calibration (Brier score, reliability curve). Results: **pending**.