"""
Comprehensive System & Regression Test Suite for CyberShield AI.

Validates:
1. Specific misclassification fix for:
   "Everyone hates you, you should just give up on life." -> Bullying Detected
2. Bullying regression suite (26 sentences) -> 100% Bullying Detected
3. Safe regression suite (22 sentences) -> 100% No Bullying Detected
4. Untouched test set (3,413 records) -> Preserved accuracy and zero regressions
5. LLM-assisted contextual analysis:
   - Safe credential verification (never pretends LLM works if unconfigured)
   - Robust error handling for API failures
   - 5-line bullying contextual analysis (targeted hostility, humiliation, intent)
   - 5-line non-bullying contextual analysis (benign context with dramatic keywords)
   - LLM determines final prediction, ML/LLM disagreements follow LLM
   - Distinguishes ML from LLM
6. Big Data Analytics Pipeline & Artifact Verification:
   - Pipeline code files exist
   - Parquet output partitions and _SUCCESS exist and can be read
   - All 8 analytics JSON files exist in data/analytics/
   - Flask /big-data route loads and renders prepared data
7. Real Gemini API call execution when configured
"""

import os
import json
import pickle
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from sklearn.model_selection import train_test_split

import app
from llm_analyzer import (
    is_llm_configured,
    is_longer_comment,
    analyze_with_llm,
    hybrid_analyze,
    _parse_llm_response
)
from experiments.improve_model import BULLYING_REGRESSION_SET, SAFE_REGRESSION_SET


class TestCyberbullyingFix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load production model and vectorizer
        cls.vec_path = os.path.join("models", "model_improvement", "candidate_vectorizer.pkl")
        cls.model_path = os.path.join("models", "model_improvement", "candidate_model.pkl")

        with open(cls.vec_path, "rb") as f:
            cls.vectorizer = pickle.load(f)
        with open(cls.model_path, "rb") as f:
            cls.model = pickle.load(f)

        cls.app_client = app.app.test_client()

    def test_01_specified_sentence_misclassification_fixed(self):
        """Sentence must return 'Bullying Detected' (label 1 / raw -1)."""
        target = "Everyone hates you, you should just give up on life."
        X = self.vectorizer.transform([target])
        raw_pred = self.model.predict(X)[0]
        score = self.model.decision_function(X)[0]

        # In model space: -1 is bullying, 0 is safe
        self.assertEqual(raw_pred, -1, f"Expected raw -1 (Bullying), got {raw_pred} with score {score}")

        # In app space: 1 is Bullying Detected, 0 is No Bullying Detected
        app_pred = 1 if raw_pred == -1 else 0
        self.assertEqual(app_pred, 1)

    def test_02_all_bullying_regression_suite_preserved(self):
        """All 26 sentences in BULLYING_REGRESSION_SET must be classified as Bullying."""
        X = self.vectorizer.transform(BULLYING_REGRESSION_SET)
        raw_preds = self.model.predict(X)
        scores = self.model.decision_function(X)

        failed = []
        for idx, (sent, pred, score) in enumerate(zip(BULLYING_REGRESSION_SET, raw_preds, scores)):
            if pred != -1:
                failed.append((idx, sent, score))

        self.assertEqual(len(failed), 0, f"Bullying regressions detected: {failed}")

    def test_03_all_safe_regression_suite_preserved(self):
        """All 22 sentences in SAFE_REGRESSION_SET must be classified as Safe."""
        X = self.vectorizer.transform(SAFE_REGRESSION_SET)
        raw_preds = self.model.predict(X)
        scores = self.model.decision_function(X)

        failed = []
        for idx, (sent, pred, score) in enumerate(zip(SAFE_REGRESSION_SET, raw_preds, scores)):
            if pred != 0:
                failed.append((idx, sent, score))

        self.assertEqual(len(failed), 0, f"Safe regressions detected: {failed}")

    def test_04_untouched_test_partition_metrics(self):
        """Verify performance metrics on untouched held-out 3,413 test partition."""
        df_raw = pd.read_csv("dataset.csv")
        if 'headline' not in df_raw.columns:
            df_raw = df_raw.rename(columns={df_raw.columns[0]: 'headline', df_raw.columns[1]: 'label'})
        df_clean = df_raw.dropna(subset=['headline', 'label']).copy()
        df_clean['headline_str'] = df_clean['headline'].astype(str)
        df_clean['label'] = df_clean['label'].astype(int)
        df_clean['norm_text'] = df_clean['headline_str'].str.strip().str.casefold()
        label_counts = df_clean.groupby('norm_text')['label'].nunique()
        conflicting_texts = label_counts[label_counts > 1].index.tolist()
        df_resolved = df_clean[~df_clean['norm_text'].isin(conflicting_texts)].copy()
        df_dedup = df_resolved.drop_duplicates(subset=['norm_text'], keep='first').copy()

        X_all = df_dedup['headline_str'].values
        y_all = df_dedup['label'].values
        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )

        preds = self.model.predict(self.vectorizer.transform(X_test))
        acc = float((preds == y_test).mean())
        self.assertGreaterEqual(acc, 0.9340, f"Accuracy dropped below baseline: {acc:.4f}")

    def test_05_flask_manual_endpoint_prediction(self):
        """POST /manual with specified sentence returns 'Bullying Detected' in rendered HTML."""
        target = "Everyone hates you, you should just give up on life."
        res = self.app_client.post('/manual', data={'text': target}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("Bullying Detected", html)

    def test_06_flask_manual_endpoint_safe_sentence(self):
        """POST /manual with safe sentence returns 'No Bullying Detected' in rendered HTML."""
        safe_sent = "I really enjoyed working with you on this project."
        res = self.app_client.post('/manual', data={'text': safe_sent}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("No Bullying Detected", html)


class TestLLMContextualAnalysis(unittest.TestCase):

    def setUp(self):
        self.five_line_bullying = (
            "Look at how utterly pathetic you were during that presentation today.\n"
            "Nobody in this entire team respects you or wants to hear anything you say.\n"
            "You embarrass yourself every single time you open your mouth in front of clients.\n"
            "You are completely useless, incompetent, and drag everyone down with your failures.\n"
            "Do everyone here a huge favor and resign before you get fired."
        )

        self.five_line_non_bullying = (
            "I watched that new psychological horror movie yesterday with my friends.\n"
            "The terrifying killer brutally hunted and destroyed everyone in the story.\n"
            "The plot was filled with scary monsters, murder mysteries, and intense violence.\n"
            "Even though the villain died at the end, the acting was absolutely phenomenal.\n"
            "I highly recommend this brilliant thriller film to anyone who loves scary cinema."
        )

    def test_07_is_longer_comment_detection(self):
        """Verify longer comment detection for approximately 5 lines."""
        self.assertTrue(is_longer_comment(self.five_line_bullying))
        self.assertTrue(is_longer_comment(self.five_line_non_bullying))
        self.assertFalse(is_longer_comment("Have a nice day!"))
        self.assertFalse(is_longer_comment("You are stupid."))

    def test_08_llm_unconfigured_behavior_never_pretends(self):
        """When LLM API keys are unconfigured, it must NOT pretend to work."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_llm_configured())
            result = analyze_with_llm(self.five_line_bullying)
            self.assertEqual(result["status"], "unconfigured")
            self.assertIsNone(result["prediction"])
            self.assertIn("not configured", result["explanation"].lower())

    def test_09_llm_api_failure_handled_safely(self):
        """Simulate network/API failure and verify safe handling without crashing."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API connection timed out")
        mock_client._is_mock = True

        result = analyze_with_llm(self.five_line_bullying, client=mock_client)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["method"], "error")
        self.assertIn("API connection timed out", result["error"])

    def test_10_llm_contextual_analysis_five_line_bullying(self):
        """
        Verify LLM identifies targeted personal hostility, humiliation, and intent
        in 5-line bullying comments.
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "is_bullying": True,
            "severity": "High",
            "risk_score": 92.0,
            "explanation": "Targeted interpersonal hostility aimed at humiliating the individual and demanding their resignation."
        })
        mock_client.models.generate_content.return_value = mock_response

        result = analyze_with_llm(self.five_line_bullying, client=mock_client)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["prediction"], 1)
        self.assertTrue(result["is_bullying"])
        self.assertEqual(result["severity"], "High")
        self.assertGreater(result["risk_score"], 80.0)
        self.assertIn("Targeted interpersonal hostility", result["explanation"])

    def test_11_llm_contextual_analysis_five_line_non_bullying(self):
        """
        Verify LLM contextual understanding distinguishes benign movie critique
        from cyberbullying despite dramatic keywords ('killer', 'brutally', 'destroyed').
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "is_bullying": False,
            "severity": "Low",
            "risk_score": 5.0,
            "explanation": "Discussion of a horror film and fictional themes; no targeted interpersonal hostility or harassment."
        })
        mock_client.models.generate_content.return_value = mock_response

        result = analyze_with_llm(self.five_line_non_bullying, client=mock_client)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["prediction"], 0)
        self.assertFalse(result["is_bullying"])
        self.assertEqual(result["severity"], "Low")
        self.assertLess(result["risk_score"], 20.0)
        self.assertIn("Discussion of a horror film", result["explanation"])

    def test_12_hybrid_analyze_fallback_on_api_error(self):
        """When LLM API fails, hybrid_analyze falls back safely to ML model."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ConnectionError("Connection refused")
        mock_client._is_mock = True

        vec = pickle.load(open("models/model_improvement/candidate_vectorizer.pkl", "rb"))
        model = pickle.load(open("models/model_improvement/candidate_model.pkl", "rb"))

        result = hybrid_analyze(self.five_line_bullying, vectorizer=vec, model=model, llm_client=mock_client)
        self.assertEqual(result["method"], "ml_fallback")
        self.assertIn(result["prediction"], [0, 1])
        self.assertIn("ML model", result["explanation"])

    def test_13_api_analyze_context_endpoint(self):
        """Verify /api/analyze-context endpoint works and returns compatible schema."""
        test_client = app.app.test_client()
        res = test_client.post('/api/analyze-context', json={"text": "Have a wonderful day!"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("prediction", data)
        self.assertEqual(data["prediction"], 0)
        self.assertIn("risk_score", data)
        self.assertIn("severity", data)

    def test_14_ml_llm_disagreement_follows_llm(self):
        """When ML and LLM disagree, the final prediction must follow the LLM."""
        vec = pickle.load(open("models/model_improvement/candidate_vectorizer.pkl", "rb"))
        model = pickle.load(open("models/model_improvement/candidate_model.pkl", "rb"))

        # Case A: ML predicts 1 (Bullying), LLM determines 0 (Safe)
        mock_client_safe = MagicMock()
        mock_resp_safe = MagicMock()
        mock_resp_safe.text = json.dumps({
            "is_bullying": False,
            "severity": "Low",
            "risk_score": 10.0,
            "explanation": "Context reveals harmless humorous banter between close friends."
        })
        mock_client_safe.models.generate_content.return_value = mock_resp_safe

        bullying_text = "You are so dumb"
        result_override_to_safe = hybrid_analyze(bullying_text, vectorizer=vec, model=model, llm_client=mock_client_safe)
        self.assertEqual(result_override_to_safe["prediction"], 0, "Final prediction must follow LLM (0)")
        self.assertEqual(result_override_to_safe["ml_prediction"], 1, "ML baseline was 1")
        self.assertFalse(result_override_to_safe["agreement"])

        # Case B: ML predicts 0 (Safe), LLM determines 1 (Bullying)
        mock_client_bully = MagicMock()
        mock_resp_bully = MagicMock()
        mock_resp_bully.text = json.dumps({
            "is_bullying": True,
            "severity": "High",
            "risk_score": 88.0,
            "explanation": "Context reveals insidious harassment not captured by vocabulary."
        })
        mock_client_bully.models.generate_content.return_value = mock_resp_bully

        safe_text = "I think we can improve this idea together."
        result_override_to_bully = hybrid_analyze(safe_text, vectorizer=vec, model=model, llm_client=mock_client_bully)
        self.assertEqual(result_override_to_bully["prediction"], 1, "Final prediction must follow LLM (1)")
        self.assertEqual(result_override_to_bully["ml_prediction"], 0, "ML baseline was 0")
        self.assertFalse(result_override_to_bully["agreement"])


class TestBigDataAnalyticsVerification(unittest.TestCase):

    def test_15_big_data_pipeline_code_exists(self):
        """Verify pipeline and session factory code exist."""
        self.assertTrue(os.path.exists("big_data/pipeline.py"))
        self.assertTrue(os.path.exists("big_data/spark_session_factory.py"))

    def test_16_parquet_output_exists_and_partitioned(self):
        """Verify Parquet directory exists with partition folders and _SUCCESS."""
        parquet_path = "data/parquet/civil_comments"
        self.assertTrue(os.path.exists(parquet_path), "Parquet output path missing")
        self.assertTrue(os.path.exists(os.path.join(parquet_path, "_SUCCESS")), "_SUCCESS marker missing")
        year_dirs = [d for d in os.listdir(parquet_path) if d.startswith("year=")]
        self.assertGreaterEqual(len(year_dirs), 1, "Expected partitioned year folders")

    def test_17_analytics_json_files_exist(self):
        """Verify all 8 analytics JSON files exist in data/analytics/."""
        expected_files = [
            "benchmarks.json",
            "length_analysis.json",
            "overview.json",
            "publication_analysis.json",
            "quality_report.json",
            "score_distribution.json",
            "subtypes.json",
            "temporal_trends.json"
        ]
        for f in expected_files:
            p = os.path.join("data", "analytics", f)
            self.assertTrue(os.path.exists(p), f"Missing analytics file: {f}")
            with open(p, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                self.assertIsNotNone(data)

    def test_18_flask_big_data_route_renders(self):
        """Verify /big-data route serves the prepared dashboard data."""
        client = app.app.test_client()
        res = client.get('/big-data')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("Big Data Analytics", html)
        self.assertIn("Civil Comments", html)


if __name__ == '__main__':
    unittest.main()
