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
            # Strip rule-override notes from display to keep PDF clean
            display_reasoning = dim_reasoning.split(' [Rule:')[0]
            truncated = display_reasoning[:120] + '…' if len(display_reasoning) > 120 else display_reasoning
            score_table_data.append([
                Paragraph(dim.replace('_', ' ').title(), styles['Normal']),
                Paragraph(f"<b>{score:.0f}/100</b>", styles['Normal']),
                Paragraph(f"{weight*100:.0f}%", styles['Normal']),
                Paragraph(truncated, styles['Normal']),
            ])

        score_table = Table(score_table_data, colWidths=[1.6*inch, 0.85*inch, 0.7*inch, 4.0*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fafafa')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Skills Analysis
        story.append(Paragraph("Skills Analysis", header_style))
        cv_lower = [s.lower() for s in cv_data.get('skills', [])]
        all_jd = list(jd_data.get('required_skills', [])) + list(jd_data.get('preferred_skills', []))

        def _pdf_skill_matched(jd_skill: str) -> bool:
            jd_l = jd_skill.lower()
            words = [w for w in jd_l.split() if len(w) >= 2]
            for cv_s in cv_lower:
                if jd_l in cv_s or cv_s in jd_l:
                    return True
                if words and any(w in cv_s for w in words):
                    return True
            return False

        matched = [s for s in all_jd if _pdf_skill_matched(s)]
        missing = [s for s in all_jd if not _pdf_skill_matched(s)]

        match_pct = round(len(matched) / len(all_jd) * 100) if all_jd else 0
        story.append(Paragraph(
            f"<b>{len(matched)} / {len(all_jd)} JD skills matched ({match_pct}%)</b>", normal_style
        ))
        story.append(Spacer(1, 0.1*inch))

        matched_text = Paragraph(
            ''.join(f'• {s}<br/>' for s in matched) or 'No skills matched',
            ParagraphStyle('matched', parent=normal_style, textColor=colors.HexColor('#155724'))
        )
        missing_text = Paragraph(
            ''.join(f'• {s}<br/>' for s in missing) or 'All skills matched!',
            ParagraphStyle('missing', parent=normal_style, textColor=colors.HexColor('#721c24'))
        )

        skills_data = [
            [Paragraph('<b>✅ Matched Skills</b>', styles['Normal']),
             Paragraph('<b>❌ Missing Skills</b>', styles['Normal'])],
            [matched_text, missing_text],
        ]

        skills_table = Table(skills_data, colWidths=[3.6*inch, 3.6*inch])
        skills_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda')),
            ('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#f8d7da')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
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