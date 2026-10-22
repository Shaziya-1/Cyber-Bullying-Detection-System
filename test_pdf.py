from fpdf import FPDF
import os

pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=15)

# Page 1: Cover Page Image
pdf.add_page()
if os.path.exists('cover_page.png'):
    pdf.image('cover_page.png', x=15, y=10, w=180)

# Page 2: Report Content
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.set_text_color(15, 23, 42)
pdf.cell(0, 10, '1. Team Details & Academic Affiliation', 0, 1, 'L')
pdf.set_font('Arial', '', 10)
pdf.multi_cell(0, 6, 'Student 1: Sajiya Shaikh (Roll No: 24102A2005)\nStudent 2: Tamanna Shaikh (Roll No: 24102A2006)\nSupervisor: Prof. Suja Jayachandran\nVidyalankar Institute of Technology (AY 2026-27)')

pdf.output('Mini_Project_Report_Cyberbullying_Detection.pdf', 'F')
print('Test PDF created successfully!')
