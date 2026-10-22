# Big Data Quality Report: Civil Comments Dataset

This report documents the empirical data quality audit conducted using **Apache PySpark (Java 21 runtime)** on the real-world Civil Comments dataset (`civil_comments_train.csv`).

## 1. Executive Quality Summary

| Metric | Measured Value | Percentage / Note |
| :--- | :--- | :--- |
| **Total Input Records** | **1,804,874** | 100.00% raw records |
| **Valid Records (Cleaned)** | **1,804,874** | 100.00% retained |
| **Invalid Records Dropped** | **0** | 0.00% (null text or target) |
| **Unique Comments** | **1,780,823** | Distinct comment texts |
| **Duplicate Comment Texts** | **24,051** | Repeated textual occurrences |
| **Exact Duplicate Rows** | **0** | Full row duplicates |
| **Toxicity Rate** | **8.0%** | Target threshold >= 0.5 |
| **Average Toxicity Score** | **0.103** | Continuous [0.0, 1.0] |

## 2. Null Value Distribution

Audit of missing/null entries across all primary schema attributes:

| Column Name | Null Count | Null Rate (%) |
| :--- | :--- | :--- |
| `id` | 0 | 0.00% |
| `target` | 0 | 0.00% |
| `comment_text` | 0 | 0.00% |
| `severe_toxicity` | 0 | 0.00% |
| `obscene` | 0 | 0.00% |
| `identity_attack` | 0 | 0.00% |
| `insult` | 0 | 0.00% |
| `threat` | 0 | 0.00% |
| `sexual_explicit` | 0 | 0.00% |
| `created_date` | 0 | 0.00% |
| `publication_id` | 0 | 0.00% |
| `article_id` | 0 | 0.00% |
| `rating` | 0 | 0.00% |

## 3. Comment Length Distribution

- **Minimum Length**: 1 characters
- **Maximum Length**: 1906 characters
- **Average Length**: 297.23 characters
- **Standard Deviation**: 269.2 characters

### Toxicity by Comment Length Segment

| Length Group | Total Comments | Toxic Comments | Toxicity Rate (%) | Avg Toxicity Score |
| :--- | :--- | :--- | :--- | :--- |
| **Short (<50 chars)** | 195,083 | 13,096 | 6.71% | 0.0777 |
| **Medium (50-200 chars)** | 702,924 | 60,619 | 8.62% | 0.1054 |
| **Long (200-500 chars)** | 556,149 | 47,425 | 8.53% | 0.1103 |
| **Very Long (>500 chars)** | 350,718 | 23,194 | 6.61% | 0.1008 |

## 4. Toxicity Subtype Distribution

| Subtype | Frequency (Count >= 0.5) | Prevalence Rate (%) | Average Score | Rank (Window) |
| :--- | :--- | :--- | :--- | :--- |
| **Insult** | 106,534 | 5.9% | 0.0812 | #1 |
| **Identity Attack** | 13,410 | 0.74% | 0.0226 | #2 |
| **Obscene** | 9,603 | 0.53% | 0.0139 | #3 |
| **Sexual Explicit** | 4,686 | 0.26% | 0.0066 | #4 |
| **Threat** | 4,280 | 0.24% | 0.0093 | #5 |
| **Severe Toxicity** | 13 | 0.0% | 0.0046 | #6 |

## 5. Toxicity Score Histogram

| Toxicity Score Bin | Comment Count | Share (%) |
| :--- | :--- | :--- |
| `[0.0 - 0.1)` | 1,266,229 | 70.16% |
| `[0.1 - 0.2)` | 160,959 | 8.92% |
| `[0.2 - 0.3)` | 111,597 | 6.18% |
| `[0.3 - 0.4)` | 64,613 | 3.58% |
| `[0.4 - 0.5)` | 57,142 | 3.17% |
| `[0.5 - 0.6)` | 47,924 | 2.66% |
| `[0.6 - 0.7)` | 37,736 | 2.09% |
| `[0.7 - 0.8)` | 27,843 | 1.54% |
| `[0.8 - 0.9)` | 21,605 | 1.20% |
| `[0.9 - 1.0]` | 9,226 | 0.51% |

---
*Report generated automatically by `big_data/pipeline.py` using Apache PySpark 4.2.0 and Eclipse Temurin OpenJDK 21.*
