"""HR Report Generator Agent."""
import logging
from typing import Dict, Any
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime

logger = logging.getLogger(__name__)

_RISK_COLOR = {"LOW": "#2e7d32", "MEDIUM": "#e65100", "HIGH": "#c62828"}


def _score_color(score: float) -> str:
    if score >= 75:
        return "#2e7d32"
    if score >= 50:
        return "#e65100"
    return "#c62828"


def generate_report(cv_data: Dict[str, Any], jd_data: Dict[str, Any], score_data: Dict[str, Any], 
                   compliance_data: Dict[str, Any], questions_data: Dict[str, Any]) -> bytes:
    """
    Generate professional PDF HR report.
    
    Args:
        cv_data: Parsed CV data
        jd_data: Analysed JD data
        score_data: Scoring results
        compliance_data: Compliance check results
        questions_data: Generated questions
        
    Returns:
        PDF content as bytes
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
        header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=18, spaceAfter=20)
        normal_style = styles['Normal']
        
        story = []
        
        # Header
        story.append(Paragraph("HireIQ — AI Recruitment Intelligence Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        story.append(Paragraph(f"Candidate: {cv_data.get('candidate_name', 'Unknown')}", normal_style))
        story.append(Paragraph(f"Role: {jd_data.get('job_title', 'Unknown')}", normal_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", header_style))
        total_score = score_data.get('total_score', 0)
        recommendation = score_data.get('recommendation', 'UNKNOWN')
        
        score_color = _score_color(total_score)
        score_text = f'<font color="{score_color}">Overall Score: {total_score}/100</font>'
        story.append(Paragraph(score_text, normal_style))
        story.append(Paragraph(f"Recommendation: <b>{recommendation}</b>", normal_style))
        
        summary = f"This candidate shows {recommendation.lower()} potential for the {jd_data.get('job_title', 'position')} role. "
        if total_score >= 75:
            summary += "Strong technical alignment and experience match."
        elif total_score >= 50:
            summary += "Some gaps that may require further assessment."
        else:
            summary += "Significant gaps in required qualifications."
        story.append(Paragraph(summary, normal_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Score Breakdown
        story.append(Paragraph("Score Breakdown", header_style))
        scores = score_data.get('scores', {})
        reasoning_map = score_data.get('reasoning', {})

        score_table_data = [['Dimension', 'Score', 'Weight', 'Reasoning']]
        for dim, score in scores.items():
            weight = score_data.get('weights', {}).get(dim, 0)
            dim_reasoning = reasoning_map.get(dim, '')
            truncated = dim_reasoning[:100] + '...' if len(dim_reasoning) > 100 else dim_reasoning
            score_table_data.append([dim.replace('_', ' ').title(), f"{score}/100", f"{weight*100:.0f}%", truncated])
        
        score_table = Table(score_table_data)
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Skills Analysis
        story.append(Paragraph("Skills Analysis", header_style))
        cv_skills = set(cv_data.get('skills', []))
        jd_required = set(jd_data.get('required_skills', []))
        jd_preferred = set(jd_data.get('preferred_skills', []))
        
        matched = cv_skills & (jd_required | jd_preferred)
        missing = (jd_required | jd_preferred) - cv_skills
        
        skills_data = [
            ['Matched Skills', 'Missing Skills'],
            ['\n'.join(matched), '\n'.join(missing)]
        ]
        
        skills_table = Table(skills_data)
        skills_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (0, 1), colors.green),
            ('BACKGROUND', (1, 1), (1, 1), colors.red),
        ]))
        story.append(skills_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Compliance & Legal
        story.append(Paragraph("Compliance & Legal Review", header_style))
        risk_level = compliance_data.get('risk_level', 'UNKNOWN')
        risk_color_hex = _RISK_COLOR.get(risk_level, "#616161")

        story.append(Paragraph(f"Risk Level: <font color='{risk_color_hex}'><b>{risk_level}</b></font>", normal_style))
        
        flags = compliance_data.get('compliance_flags', [])
        if flags:
            story.append(Paragraph("Compliance Flags:", styles['Heading3']))
            for flag in flags:
                story.append(Paragraph(f"• {flag}", normal_style))
        
        gdpr = compliance_data.get('gdpr_requirements', [])
        if gdpr:
            story.append(Paragraph("GDPR Requirements:", styles['Heading3']))
            for req in gdpr:
                story.append(Paragraph(f"• {req}", normal_style))
        
        story.append(Spacer(1, 0.25*inch))
        
        # Interview Questions
        story.append(Paragraph("Recommended Interview Questions", header_style))
        questions = questions_data.get('questions', [])
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"{i}. {q.get('question_text', '')}", styles['Heading4']))
            story.append(Paragraph(f"<i>What to listen for:</i> {q.get('what_to_listen_for', '')}", normal_style))
            story.append(Paragraph(f"<i>Red flags:</i> {q.get('red_flag_indicators', '')}", normal_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Disclaimer: This report is generated by AI and should be used as a screening tool only. Final hiring decisions should involve human review and additional verification.", 
                              ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, textColor=colors.grey)))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return b"Error generating PDF"