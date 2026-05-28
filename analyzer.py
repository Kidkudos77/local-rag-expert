"""
analyzer.py — Direct LLM analysis, literature review, glossary, research gaps,
               cross-paper comparison, and multimodal image analysis.
"""

import os
import base64
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq

LLM_MODEL    = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"

PROMPTS = {
    "Title": """You are a research assistant helping a graduate student understand academic papers.
Analyze the following paper title and explain in plain English:
1. What topic or problem the paper is about
2. Who the likely subjects or context are (if mentioned)
3. What kind of study this appears to be
4. Any important keywords or technical terms and what they mean

Title: {text}""",

    "Abstract": """You are a research assistant helping a graduate student understand academic papers.
Break down the following abstract in plain English:
1. What problem or question the study addresses
2. What methods were used
3. What the main findings were
4. What the authors concluded
5. Why this research matters

Abstract: {text}""",

    "Introduction": """You are a research assistant helping a graduate student understand academic papers.
Analyze the following introduction and explain:
1. What background context is being established
2. What gap in knowledge the authors identified
3. What the research question or hypothesis is
4. How this study fits into the existing literature

Introduction: {text}""",

    "Methods": """You are a research assistant helping a graduate student understand academic papers.
Break down the following methods section:
1. What type of study design was used
2. Who or what the subjects/samples were
3. What procedures or techniques were applied
4. What was measured and how
5. Any potential weaknesses or limitations

Methods: {text}""",

    "Results": """You are a research assistant helping a graduate student understand academic papers.
Analyze the following results section:
1. What the main findings were in plain language
2. Whether the results supported or contradicted the hypothesis
3. Any surprising or noteworthy patterns
4. What the numbers or statistics mean in practical terms

Results: {text}""",

    "Discussion / Conclusion": """You are a research assistant helping a graduate student understand academic papers.
Analyze the following discussion or conclusion:
1. How the authors interpreted their findings
2. Whether the interpretations are well-supported
3. What limitations the authors acknowledged
4. What future research directions are suggested
5. What the real-world implications are

Discussion/Conclusion: {text}""",

    "Custom section": """You are a research assistant helping a graduate student understand academic papers.
Analyze the following text and provide:
1. A plain-English summary
2. An explanation of any technical terms or jargon
3. The key takeaway or main point
4. Questions a critical reader should ask

Text: {text}""",
}


def _llm_chain(prompt_template: str, text: str) -> str:
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm    = ChatGroq(model=LLM_MODEL)
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


# ── Section analysis ──────────────────────────────────────────────────────────

def analyze_text(text: str, section_type: str) -> str:
    template = PROMPTS.get(section_type, PROMPTS["Custom section"])
    return _llm_chain(template, text)


# ── Paper summary ─────────────────────────────────────────────────────────────

def summarize_paper(full_text: str, filename: str) -> str:
    excerpt  = full_text[:4000]
    template = """You are a research assistant. Based on the following excerpt, write a plain-English abstract (4-6 sentences) covering:
1. What problem the paper addresses
2. What approach or method was used
3. What the main findings or contributions are
4. Why it matters

Write as a paragraph, not a list. Be concise and clear.

Paper excerpt:
{text}

Plain-English Abstract:"""
    return _llm_chain(template, excerpt)


# ── Literature review generator ───────────────────────────────────────────────

def generate_literature_review(summaries: dict) -> str:
    """
    summaries: {filename: summary_text}
    Generates a structured 3-4 paragraph literature review.
    """
    combined = "\n\n".join([
        f"Paper: {os.path.basename(k)}\n{v}"
        for k, v in summaries.items()
    ])
    template = """You are an academic writing assistant helping a graduate student write a literature review.

Based on the following paper summaries, write a structured literature review (3-4 paragraphs) that:
1. Introduces the research area and its importance
2. Groups the papers by theme, approach, or chronology
3. Identifies agreements, contradictions, and gaps across the papers
4. Concludes with what remains unknown or understudied

Write in formal academic prose. Reference specific papers by their filename when making claims.

Paper summaries:
{text}

Literature Review:"""
    return _llm_chain(template, combined)


# ── Glossary builder ──────────────────────────────────────────────────────────

def build_glossary(full_texts: dict) -> str:
    """
    full_texts: {filename: full_text}
    Extracts and defines all technical terms across all papers.
    """
    combined = "\n\n".join([
        f"From {os.path.basename(k)}:\n{v[:2000]}"
        for k, v in full_texts.items()
    ])
    template = """You are a research assistant building a glossary for a graduate student.

Extract all technical terms, acronyms, and domain-specific vocabulary from the following text.
For each term provide a brief plain-English definition (1-2 sentences).
Format each entry as: **Term**: Definition

Focus on terms that a reader unfamiliar with the field would need defined.
Group related terms together.

Text:
{text}

Glossary:"""
    return _llm_chain(template, combined)


# ── Research gap identifier ───────────────────────────────────────────────────

def identify_research_gaps(summaries: dict) -> str:
    """
    Analyzes multiple paper summaries and identifies research gaps.
    """
    combined = "\n\n".join([
        f"Paper: {os.path.basename(k)}\n{v}"
        for k, v in summaries.items()
    ])
    template = """You are a research assistant helping a graduate student identify gaps in the literature.

Based on the following paper summaries, identify:
1. **Questions left unanswered** — what do these papers fail to address?
2. **Methodological limitations** — what approaches haven't been tried?
3. **Underrepresented contexts** — what populations, settings, or domains are missing?
4. **Contradictions needing resolution** — where do papers disagree without resolution?
5. **Future research opportunities** — what are the most promising directions?

Be specific and reference individual papers when relevant.

Paper summaries:
{text}

Research Gaps Analysis:"""
    return _llm_chain(template, combined)


# ── Cross-paper comparison ────────────────────────────────────────────────────

def compare_papers(question: str, full_texts: dict) -> str:
    """
    Directly compares papers on a specific question without RAG retrieval.
    Uses first 2000 chars of each paper for speed.
    """
    combined = "\n\n".join([
        f"=== {os.path.basename(k)} ===\n{v[:2000]}"
        for k, v in full_texts.items()
    ])
    template = """You are a research assistant specializing in comparative analysis of academic papers.

Compare the following papers specifically on this question: {question}

For each paper, note what it says (or doesn't say) about the topic.
Then provide a synthesis highlighting similarities, differences, and contradictions.

Papers:
{text}

Comparative Analysis:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm    = ChatGroq(model=LLM_MODEL)
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "text": combined})


# ── Multimodal image analysis ─────────────────────────────────────────────────

def analyze_image(image_bytes: bytes, question: str) -> str:
    """
    Send an image (figure/chart from a paper) to Groq's vision model for analysis.
    """
    client    = Groq()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": question or "Describe this figure from an academic paper in plain English. Explain what the data shows and what conclusions can be drawn."
                }
            ]
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content
