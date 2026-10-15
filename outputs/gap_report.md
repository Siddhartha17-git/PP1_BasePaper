# OBE Knowledge Gap Engine Report

## Evaluation Parameters
- **Target Learner ID**: `78784` (Held-out Test Split)
- **Total Historical Interactions**: 87
- **Mastery Threshold (Beginner cutoff)**: 50.0%
- **Mastery Target (Proficiency cutoff)**: 70.0%
- **Gap Condition**: Any concept with Mastery < Target (70.0%) is flagged for remedial recommendation.

---

## 1. Paper Demonstration Vector Verification
Vector provided in base paper: `[0.82, 0.35, 0.76, 0.60, 0.91]`

| Concept ID | Synthetic Concept Name | Mastery Score | OBE Level | Gap Status | Action Triggered |
|---|---|---|---|---|---|
| C1 | Concept 1 (Foundations) | 82.0% | **Expert** | Proficient | No Action |
| C2 | Concept 2 (Core Mechanics) | 35.0% | **Beginner** | **FLAGGED GAP** | Send to Remediation |
| C3 | Concept 3 (Problem Solving) | 76.0% | **Expert** | Proficient | No Action |
| C4 | Concept 4 (Advanced Synthesis) | 60.0% | **Intermediate** | **FLAGGED GAP** | Send to Remediation |
| C5 | Concept 5 (Real-World Application) | 91.0% | **Expert** | Proficient | No Action |

---

## 2. Real Held-Out Learner Knowledge State Evaluation
Evaluated using the trained EKT knowledge tracing model on learner `#78784`.

| Concept Index | Concept Name | EKT Mastery Probability | OBE Classification | Remediation Flag | Learner History (Correct / Total) |
|---|---|---|---|---|---|
| 4 | Counting Methods | 52.08% | **Intermediate** | **FLAGGED FOR REMEDIATION** | 0 / 0 |
| 19 | Write Linear Equation from Graph | 54.98% | **Intermediate** | **FLAGGED FOR REMEDIATION** | 0 / 0 |
| 7 | Finding Percents | 65.18% | **Intermediate** | **FLAGGED FOR REMEDIATION** | 0 / 0 |
| 2 | Circle Graph | 67.69% | **Intermediate** | **FLAGGED FOR REMEDIATION** | 0 / 0 |
| 10 | Order of Operations All | 73.61% | **Expert** | Proficient (Expert) | 0 / 0 |
| 15 | Pythagorean Theorem | 75.10% | **Expert** | Proficient (Expert) | 0 / 0 |
| 14 | Proportion | 77.21% | **Expert** | Proficient (Expert) | 0 / 0 |
| 5 | Equation Solving More Than Two Steps | 79.97% | **Expert** | Proficient (Expert) | 0 / 0 |
| 18 | Venn Diagram | 80.71% | **Expert** | Proficient (Expert) | 0 / 0 |
| 16 | Subtraction Whole Numbers | 81.26% | **Expert** | Proficient (Expert) | 8 / 8 |
| 3 | Conversion of Fraction Decimals Percents | 82.55% | **Expert** | Proficient (Expert) | 0 / 0 |
| 0 | Addition and Subtraction Fractions | 85.92% | **Expert** | Proficient (Expert) | 21 / 25 |
| 6 | Equation Solving Two or Fewer Steps | 87.18% | **Expert** | Proficient (Expert) | 0 / 0 |
| 17 | Table | 90.41% | **Expert** | Proficient (Expert) | 0 / 0 |
| 1 | Addition and Subtraction Integers | 92.58% | **Expert** | Proficient (Expert) | 11 / 12 |
| 12 | Ordering Positive Decimals | 93.08% | **Expert** | Proficient (Expert) | 6 / 6 |
| 8 | Multiplication Fractions | 93.90% | **Expert** | Proficient (Expert) | 16 / 19 |
| 13 | Probability of a Single Event | 95.09% | **Expert** | Proficient (Expert) | 0 / 0 |
| 11 | Ordering Fractions | 97.07% | **Expert** | Proficient (Expert) | 3 / 3 |
| 9 | Multiplication and Division Integers | 97.16% | **Expert** | Proficient (Expert) | 14 / 14 |

---

## 3. Summary & Remedial Routing
- **Total Tracked Concepts**: 20
- **Expert Concepts (Mastery > 70%)**: 16
- **Concepts Requiring Remediation (Mastery < 70%)**: 4
  - **Beginner Concepts (< 50%)**: 0
  - **Intermediate Concepts (50% - 70%)**: 4
- **Top Priority Concept for Remediation**: `Counting Methods` (Mastery: 52.08%)
