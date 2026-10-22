from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 8, 'Cyberbullying Detection & Big Data Analytics System | Mini Project Report', 0, 0, 'L')
            self.cell(0, 8, 'VIT (AY 2026-27)', 0, 1, 'R')
            self.set_draw_color(203, 213, 225)
            self.line(10, 16, 200, 16)
            self.ln(4)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-14)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(148, 163, 184)
            self.cell(0, 8, f'Page {self.page_no()} | Department of Computer Engineering, VIT', 0, 0, 'C')

pdf = PDFReport(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=16)

# ==========================================================
# PAGE 1: COVER PAGE (Exact User Attached Image)
# ==========================================================
pdf.add_page()
if os.path.exists('cover_page.png'):
    pdf.image('cover_page.png', x=12, y=10, w=186)

# ==========================================================
# PAGE 2: SECTIONS 1, 2, 3, 4
# ==========================================================
pdf.add_page()

def print_section(title):
    pdf.set_font('Arial', 'B', 13)
    pdf.set_text_color(15, 23, 42)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 8, f"  {title}", 0, 1, 'L', fill=True)
    pdf.ln(2)

def print_subsection(title):
    pdf.set_font('Arial', 'B', 10.5)
    pdf.set_text_color(2, 132, 199)
    pdf.cell(0, 6, title, 0, 1, 'L')
    pdf.ln(1)

def print_p(text, prefix=""):
    pdf.set_font('Arial', '', 9.5)
    pdf.set_text_color(51, 65, 85)
    full_text = f"{prefix} {text}" if prefix else text
    pdf.multi_cell(0, 4.8, full_text)
    pdf.ln(1.5)

def print_bullet(title, desc):
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, f"  * {title}: {desc}")
    pdf.ln(1)

# 1. Team Details
print_section("1. Team Details & Academic Affiliation")
pdf.set_font('Arial', 'B', 8.5)
pdf.set_fill_color(30, 58, 138)
pdf.set_text_color(255, 255, 255)
pdf.cell(45, 6.5, " Role / Designation", 1, 0, 'L', fill=True)
pdf.cell(75, 6.5, " Name / Roll Number", 1, 0, 'L', fill=True)
pdf.cell(70, 6.5, " Academic Department", 1, 1, 'L', fill=True)

team_rows = [
    ("Student 1", "Sajiya Shaikh (Roll No: 24102A2005)", "Computer Engineering, VIT"),
    ("Student 2", "Tamanna Shaikh (Roll No: 24102A2006)", "Computer Engineering, VIT"),
    ("Project Supervisor", "Prof. Suja Jayachandran", "Faculty Guide, Computer Engg."),
    ("Institution & Session", "Vidyalankar Institute of Technology", "NAAC A+ | AY 2026-27 | Subject: NLP")
]

pdf.set_font('Arial', '', 8.5)
for i, (r1, r2, r3) in enumerate(team_rows):
    if i % 2 == 1:
        pdf.set_fill_color(248, 250, 252)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(45, 5.8, f" {r1}", 1, 0, 'L', fill=True)
    pdf.cell(75, 5.8, f" {r2}", 1, 0, 'L', fill=True)
    pdf.cell(70, 5.8, f" {r3}", 1, 1, 'L', fill=True)
pdf.ln(3)

# 2. Title of the Project
print_section("2. Title of the Project")
print_p("CYBERBULLYING DETECTION & BIG DATA ANALYTICS SYSTEM", prefix="Project Title:")
print_p("An end-to-end natural language processing and distributed big data platform for automated cyber harassment identification, live case moderation, and longitudinal toxicity pattern analytics.", prefix="Scope & Focus:")
pdf.ln(1)

# 3. Detailed Problem Statement
print_section("3. Detailed Problem Statement")
print_p("Modern digital discourse across social networking platforms is plagued by abusive commentary, personal attacks, hate speech, and cyberbullying. Platform operators face critical technical and operational bottlenecks:")
print_bullet("Scalability Bottleneck", "Human moderators cannot manually review millions of incoming comments daily without latency spikes and high operational overhead.")
print_bullet("Adversarial & Nuanced Text", "Offenders deliberately employ obfuscated spelling, intentional leetspeak, sarcastic framing, and colloquial phrases to bypass simple keyword blacklists.")
print_bullet("Class Imbalance", "Only 5-10% of typical web comments are toxic. Models trained on skewed data either trigger massive false positives or fail to recall dangerous threats.")
print_bullet("Disjoint Systems", "Traditional architectures lack synchronization between real-time low-latency inference (<50ms) and large-scale big data historical analysis.")
pdf.ln(2)

# 4. Technology/Algorithm Used
print_section("4. Technology / Algorithm Used")
print_subsection("4.1 Natural Language Processing & Machine Learning")
print_bullet("Preprocessing Pipeline", "Normalizing text casing, stripping URLs and regex special symbols, and applying custom stopword filters.")
print_bullet("FeatureUnion TF-IDF", "Combining Word N-Grams (1-2) with Character N-Grams (3-5) via scikit-learn FeatureUnion. Character n-grams provide crucial resilience against misspelled slurs and masked tokens.")
print_bullet("SGDClassifier (Linear SVM)", "Trained with hinge loss (linear SVM). Best overall model achieving 92.44% Accuracy and 92.48% F1-score with ultra-fast sub-10ms inference.")
print_bullet("Logistic Regression", "Provides calibrated class probabilities, yielding 91.89% Accuracy and 97.10% ROC-AUC for continuous risk ranking.")
print_bullet("Comparative Algorithms", "LinearSVC (91.33% Acc), Decision Tree (91.86% Acc), Naive Bayes (86.94% Acc), and AdaBoost (80.05% Acc).")

print_subsection("4.2 Distributed Big Data Engineering")
print_bullet("Apache PySpark 4.2.0", "Distributed pipeline executing on Java 21 runtime managing 1.8M comments with distributed transformations.")
print_bullet("Snappy Parquet Storage", "Columnar Snappy Parquet storage achieving a 94.6% space reduction (41.8 MB vs 778.4 MB) and 17.9x query speedup.")
print_bullet("Spark Window Functions", "Dense ranking of toxic sources via dense_rank() and cumulative longitudinal growth tracking.")

print_subsection("4.3 Web Platform & Moderation Architecture")
print_bullet("Flask 3.1.3 Web App", "Modular microservice backend with SQLite case database, live simulation, and active moderation management.")
pdf.ln(2)

# Architecture figure
if os.path.exists('report_assets/fig4_architecture_pipeline.png'):
    pdf.image('report_assets/fig4_architecture_pipeline.png', x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 94)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, 'Figure 1: End-to-End System Pipeline Architecture - Ingestion, FeatureUnion TF-IDF, PySpark & Moderation UI', 0, 1, 'C')
    pdf.ln(3)

# ==========================================================
# PAGE 3: DATASET & RESULTS
# ==========================================================
pdf.add_page()
print_section("5. Dataset Description")
print_p("The framework was built and evaluated using two complementary data tiers to validate both real-time ML inference and distributed big data scalability:")

pdf.set_font('Arial', 'B', 8)
pdf.set_fill_color(2, 132, 199)
pdf.set_text_color(255, 255, 255)
pdf.cell(48, 6.5, " Dataset Name", 1, 0, 'L', fill=True)
pdf.cell(32, 6.5, " Volume / Size", 1, 0, 'L', fill=True)
pdf.cell(55, 6.5, " Target Classes & Features", 1, 0, 'L', fill=True)
pdf.cell(55, 6.5, " Operational Role in System", 1, 1, 'L', fill=True)

data_table = [
    ("Core Cyberbullying (dataset.csv)", "17,065 Clean Rows\n4.91 MB", "Binary: Bullying (-1) [62.0%]\nSafe (0) [38.0%]", "Supervised training & clean zero-leakage evaluation."),
    ("Civil Comments Corpus (civil_comments.csv)", "1,804,874 Rows\n778.4 MB (41.8 MB Pq)", "Continuous Toxicity [0,1],\nInsult, Threat, Obscene", "PySpark big data analytics & window function metrics."),
    ("Instagram Comments (instagram_comments.csv)", "82.75 MB Raw Text", "Unstructured Social Comments\nColloquial Text & Slang", "Simulation validation & robustness testing against slang.")
]

pdf.set_font('Arial', '', 8)
for i, (d1, d2, d3, d4) in enumerate(data_table):
    if i % 2 == 1:
        pdf.set_fill_color(248, 250, 252)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(48, 6, f" {d1[:28]}", 1, 0, 'L', fill=True)
    pdf.cell(32, 6, f" {d2.splitlines()[0]}", 1, 0, 'L', fill=True)
    pdf.cell(55, 6, f" {d3.splitlines()[0]}", 1, 0, 'L', fill=True)
    pdf.cell(55, 6, f" {d4[:32]}...", 1, 1, 'L', fill=True)
pdf.ln(3)

print_section("6. Screenshot / Results & Benchmarks")
print_subsection("6.1 Model Performance Evaluation on Unseen Test Set (N=3,413)")

pdf.set_font('Arial', 'B', 8)
pdf.set_fill_color(30, 41, 59)
pdf.set_text_color(255, 255, 255)
pdf.cell(50, 6, " Algorithm", 1, 0, 'L', fill=True)
pdf.cell(28, 6, " Accuracy", 1, 0, 'C', fill=True)
pdf.cell(28, 6, " Precision", 1, 0, 'C', fill=True)
pdf.cell(28, 6, " Recall", 1, 0, 'C', fill=True)
pdf.cell(28, 6, " F1-Score", 1, 0, 'C', fill=True)
pdf.cell(28, 6, " ROC-AUC", 1, 1, 'C', fill=True)

bench_rows = [
    ("SGDClassifier (Best Linear SVM)", "92.44%", "92.63%", "92.44%", "92.48%", "N/A"),
    ("Logistic Regression", "91.89%", "92.05%", "91.89%", "91.93%", "97.10%"),
    ("Decision Tree Classifier", "91.86%", "91.87%", "91.86%", "91.86%", "91.75%"),
    ("Linear SVC", "91.33%", "91.48%", "91.33%", "91.37%", "N/A"),
    ("Multinomial Naive Bayes", "86.94%", "87.29%", "86.94%", "86.61%", "95.01%"),
    ("AdaBoost Classifier", "80.05%", "85.60%", "80.05%", "80.28%", "87.59%")
]

pdf.set_font('Arial', '', 8)
for i, (b1, b2, b3, b4, b5, b6) in enumerate(bench_rows):
    is_top = (i == 0)
    if is_top:
        pdf.set_fill_color(236, 253, 245)
        pdf.set_text_color(6, 95, 70)
        pdf.set_font('Arial', 'B', 8)
    else:
        if i % 2 == 1:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font('Arial', '', 8)
    pdf.cell(50, 5.5, f" {b1}", 1, 0, 'L', fill=True)
    pdf.cell(28, 5.5, b2, 1, 0, 'C', fill=True)
    pdf.cell(28, 5.5, b3, 1, 0, 'C', fill=True)
    pdf.cell(28, 5.5, b4, 1, 0, 'C', fill=True)
    pdf.cell(28, 5.5, b5, 1, 0, 'C', fill=True)
    pdf.cell(28, 5.5, b6, 1, 1, 'C', fill=True)
pdf.ln(3)

# Figures 1 & 2
if os.path.exists('report_assets/fig1_model_performance.png'):
    pdf.image('report_assets/fig1_model_performance.png', x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 92)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, 'Figure 2: Empirical Benchmark Comparison across Evaluated NLP Classifiers', 0, 1, 'C')
    pdf.ln(4)

# ==========================================================
# PAGE 4: CONFUSION MATRICES & BIG DATA BENCHMARKS
# ==========================================================
pdf.add_page()

if os.path.exists('report_assets/fig2_confusion_matrices.png'):
    pdf.image('report_assets/fig2_confusion_matrices.png', x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 80)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, 'Figure 3: Confusion Matrices for SGDClassifier (Best Overall) and Logistic Regression', 0, 1, 'C')
    pdf.ln(3)

print_subsection("6.2 PySpark Big Data Distributed Performance (1.8M Records)")
print_p("PySpark parallel processing achieved an overall throughput of ~7,267 records/sec. Parquet columnar compression shrank data volume from 778.4 MB to 41.83 MB (94.6% reduction), while query read latency dropped from 18.17 s to 1.015 s (17.9x speedup).")

if os.path.exists('report_assets/fig3_bigdata_analytics.png'):
    pdf.image('report_assets/fig3_bigdata_analytics.png', x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 120)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, 'Figure 4: Big Data PySpark Analytics - Class Split, Harassment Subtypes, Parquet Storage & Query Latency', 0, 1, 'C')
    pdf.ln(3)

# ==========================================================
# PAGE 5: WEB UI, CONCLUSION & GITHUB LINK
# ==========================================================
pdf.add_page()
print_subsection("6.3 Web Interface & Live Moderation Simulation")
print_p("The Flask application offers a real-time testing playground, risk scoring (85% High Severity vs 10% Low Severity), automated persistent auditing in SQLite, and dynamic administrative escalation.")

if os.path.exists('report_assets/fig5_web_interface_simulation.png'):
    pdf.image('report_assets/fig5_web_interface_simulation.png', x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 94)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, 'Figure 5: CyberShield AI Web Platform - Real-time NLP Detection & Moderation Case Queue', 0, 1, 'C')
    pdf.ln(4)

# 7. Conclusion
print_section("7. Conclusion & Future Scope")
print_bullet("High Accuracy & F1-Score", "The SGDClassifier linear SVM model reached 92.44% Accuracy, 92.63% Precision, and 92.48% F1-score with 0.00% data leakage on held-out test data.")
print_bullet("Adversarial Robustness", "FeatureUnion character n-grams provided crucial resilience against adversarial misspellings and masked expletives.")
print_bullet("Distributed Scalability", "Snappy Parquet columnar encoding reduced storage by 94.6% and delivered 17.9x faster query read speeds across 1.8M records.")
print_bullet("Operational Value", "Interactive Flask interface with automated case escalation ensures actionable human-in-the-loop governance.")
print_bullet("Future Scope", "Extending to multi-modal content (images/memes) and fine-tuning transformer backbones (RoBERTa/DeBERTa) for multi-lingual coverage.")
pdf.ln(2)

# 8. GitHub Link
print_section("8. GitHub Project Repository & Code Link")
print_p("All source code, trained model artifacts, distributed PySpark scripts, datasets, and web application templates are publicly accessible at:")

pdf.set_font('Arial', 'B', 11)
pdf.set_text_color(2, 132, 199)
pdf.cell(0, 7, "https://github.com/Shaziya-1/Cyber-Bullying-Detection-System", 0, 1, 'C', link="https://github.com/Shaziya-1/Cyber-Bullying-Detection-System")
pdf.ln(2)

pdf.set_font('Arial', 'B', 8.5)
pdf.set_fill_color(30, 58, 138)
pdf.set_text_color(255, 255, 255)
pdf.cell(60, 6, " Parameter", 1, 0, 'L', fill=True)
pdf.cell(130, 6, " Detail / Resource Location", 1, 1, 'L', fill=True)

repo_info = [
    ("Public Repository", "https://github.com/Shaziya-1/Cyber-Bullying-Detection-System"),
    ("Web Application Entry", "app.py (Flask Web Application with /manual, /dashboard, /big-data)"),
    ("Model Artifacts", "models/ (best_model.pkl, best_vectorizer.pkl, candidate_model.pkl)"),
    ("Big Data Processing", "big_data/ (PySpark distributed transformations & parquet partitions)")
]

pdf.set_font('Arial', '', 8.5)
for i, (p1, p2) in enumerate(repo_info):
    if i % 2 == 1:
        pdf.set_fill_color(248, 250, 252)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5.8, f" {p1}", 1, 0, 'L', fill=True)
    pdf.cell(130, 5.8, f" {p2}", 1, 1, 'L', fill=True)

pdf.output("Mini_Project_Report_Cyberbullying_Detection.pdf", "F")
print("Saved PDF successfully as Mini_Project_Report_Cyberbullying_Detection.pdf")
