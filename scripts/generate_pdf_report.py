"""PDF Academic Report Generator for VisionQC.

Compiles a formatted project report based on the VITyarthi Computer Vision
curriculum evaluation requirements using fpdf2.
"""

from typing import List, Tuple
import os
import sys
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class VisionQCReportDocument(FPDF):
    """Publication-styled PDF document generator."""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(
                0,
                8,
                "VisionQC: Automated Industrial Defect Detection | Project Report",
                border="B",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            self.ln(5)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

    def chapter_heading(self, index: int, label: str):
        self.set_font("Helvetica", "B", 12.5)
        self.set_fill_color(238, 242, 248)
        self.set_text_color(20, 45, 85)
        self.cell(0, 8, f"{index}. {label}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(2)

    def subhead(self, text: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 70, 130)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(1)

    def paragraph(self, content: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, content)
        self.ln(2)

    def code_listing(self, code: str):
        self.set_font("Courier", "", 7.5)
        self.set_fill_color(246, 248, 250)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 3.8, code, fill=True, border=1)
        self.ln(2)


def generate_pdf_report(destination_pdf: str, project_root: str) -> None:
    """Renders the comprehensive project report."""
    doc = VisionQCReportDocument(orientation="P", unit="mm", format="A4")
    doc.set_auto_page_break(auto=True, margin=14)
    doc.alias_nb_pages()

    # Cover Page
    doc.add_page()
    doc.set_fill_color(25, 45, 80)
    doc.rect(0, 0, 210, 40, "F")

    doc.set_y(10)
    doc.set_font("Helvetica", "B", 18)
    doc.set_text_color(255, 255, 255)
    doc.cell(0, 9, "VITyarthi - Build Your Own Project", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    doc.set_font("Helvetica", "I", 10.5)
    doc.cell(0, 6, "Computer Vision Flipped Course Evaluation", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    doc.set_y(52)
    doc.set_font("Helvetica", "B", 20)
    doc.set_text_color(20, 50, 100)
    doc.multi_cell(0, 8.5, "VisionQC: Automated Industrial Defect Detection & Quality Inspection Pipeline", align="C")

    doc.ln(3)
    doc.set_font("Helvetica", "I", 10.5)
    doc.set_text_color(80, 80, 80)
    doc.cell(0, 6, "A Modular, Headless Computer Vision System for High-Throughput Manufacturing", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    doc.ln(8)
    doc.set_fill_color(248, 250, 252)
    doc.set_draw_color(200, 210, 225)
    doc.rect(20, 82, 170, 80, "DF")

    doc.set_y(86)
    meta = [
        ("Course Name:", "Computer Vision"),
        ("Target Domain:", "Industrial Automation & Quality Control"),
        ("Student Name:", "Nagahimesh Vuppala"),
        ("Registration No:", "24BAI10006"),
        ("Course Slot / Code:", "CSE3002 / Flipped Classroom Evaluation"),
        ("Submission Date:", "September 18, 2026"),
        ("GitHub Repository:", "https://github.com/Hello-himesh/visionqc"),
    ]
    for key, val in meta:
        doc.set_x(25)
        doc.set_font("Helvetica", "B", 9.5)
        doc.set_text_color(40, 40, 40)
        doc.cell(50, 6.5, key)
        doc.set_font("Helvetica", "", 9.5)
        doc.set_text_color(20, 50, 100)
        doc.cell(105, 6.5, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Core Sections
    sections = [
        ("Introduction", "VisionQC is an automated end-to-end computer vision quality inspection system engineered to deliver deterministic, high-throughput component inspection on industrial assembly lines. By combining classical image enhancement, mathematical morphology, geometric edge and corner profiling, marker-controlled watershed segmentation, and statistical learning, VisionQC autonomously evaluates manufactured parts passing along a conveyor system."),
        ("Problem Statement", "High-speed mass manufacturing operations require automated visual verification of components to eliminate subjectivity, fatigue, and latency. VisionQC tackles micro-defect isolation, concentric bore measurement, multi-part separation, and real-time inference under 150 ms per part."),
        ("Functional Requirements", "FR-1: Image Acquisition & Preprocessing (BGR/GRAY/HSV/LAB conversions, Gaussian/Median/Bilateral filters, Power-Law and Logarithmic transformations, CLAHE).\nFR-2: Morphological Defect Isolation (Erosion, Dilation, Opening, Closing, Gradient, Top-Hat, Black-Hat).\nFR-3: Geometric & Structural Validation (Sobel gradients, Canny edge detection, Harris and Shi-Tomasi corners, Hough Lines and Circles).\nFR-4: Segmentation & Clustering (Marker-Controlled Watershed, K-Means and K-Medoids clustering).\nFR-5: Machine Learning Classification & Tracking (HOG descriptors, moments, PCA SVD reduction, distance-weighted KNN, Gaussian Naive Bayes, Centroid tracking)."),
        ("Curriculum Concept Mapping", "Module 1: Color conversions and image matrix operations.\nModule 2: Denoising, point transformations, morphology, and histogram equalization.\nModule 3: Sobel gradients, Canny hysteresis, Harris/Shi-Tomasi keypoints, Hough line and circle transforms.\nModule 4: Distance-transform marker watershed and color space clustering.\nModule 5: Principal Component Analysis, K-Nearest Neighbors, Naive Bayes, and Centroid object tracking."),
        ("System Architecture & Verification", "The pipeline processes raw camera streams into conditioned matrices, applies morphological defect filtering and geometric feature extraction, performs segmentation and PCA projection, and produces quantitative JSON records alongside diagnostic visual overlays."),
    ]

    for idx, (title, body) in enumerate(sections, start=2):
        doc.add_page()
        doc.chapter_heading(idx, title)
        doc.paragraph(body)

    out_folder = os.path.dirname(destination_pdf)
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
    doc.output(destination_pdf)
    print(f"Report PDF generated successfully: {destination_pdf}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    target = os.path.join(root_dir, "VisionQC_Project_Report.pdf")
    generate_pdf_report(target, root_dir)
