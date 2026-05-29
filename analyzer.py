"""
analyzer.py — All LLM-powered analysis tools.
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
2. What kind of study this appears to be
3. Any important keywords or technical terms and what they mean
Title: {text}""",

    "Abstract": """You are a research assistant helping a graduate student understand academic papers.
Break down the following abstract in plain English:
1. What problem the study addresses
2. What methods were used
3. What the main findings were
4. What the authors concluded
5. Why this research matters
Abstract: {text}""",

    "Introduction": """You are a research assistant helping a graduate student understand academic papers.
Analyze the following introduction:
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


def _run(template: str, **kwargs) -> str:
    prompt = ChatPromptTemplate.from_template(template)
    llm    = ChatGroq(model=LLM_MODEL)
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke(kwargs)


def analyze_text(text: str, section_type: str) -> str:
    template = PROMPTS.get(section_type, PROMPTS["Custom section"])
    return _run(template, text=text)


def summarize_paper(full_text: str, filename: str) -> str:
    return _run("""You are a research assistant. Based on the following excerpt, write a plain-English abstract (4-6 sentences) covering:
1. What problem the paper addresses
2. What approach or method was used
3. What the main findings or contributions are
4. Why it matters
Write as a paragraph, not a list.
Paper excerpt: {text}
Plain-English Abstract:""", text=full_text[:4000])


def generate_literature_review(summaries: dict) -> str:
    combined = "\n\n".join([f"Paper: {os.path.basename(k)}\n{v}" for k, v in summaries.items()])
    return _run("""You are an academic writing assistant helping a graduate student write a literature review.
Based on the following paper summaries, write a structured literature review (3-4 paragraphs) that:
1. Introduces the research area and its importance
2. Groups the papers by theme, approach, or chronology
3. Identifies agreements, contradictions, and gaps across the papers
4. Concludes with what remains unknown or understudied
Write in formal academic prose. Reference specific papers by filename when making claims.
Paper summaries: {text}
Literature Review:""", text=combined)


def build_glossary(full_texts: dict) -> str:
    combined = "\n\n".join([f"From {os.path.basename(k)}:\n{v[:2000]}" for k, v in full_texts.items()])
    return _run("""You are a research assistant building a glossary for a graduate student.
Extract all technical terms, acronyms, and domain-specific vocabulary from the following text.
For each term provide a brief plain-English definition (1-2 sentences).
Format each entry as: **Term**: Definition
Focus on terms that a reader unfamiliar with the field would need defined.
Text: {text}
Glossary:""", text=combined)


def identify_research_gaps(summaries: dict) -> str:
    combined = "\n\n".join([f"Paper: {os.path.basename(k)}\n{v}" for k, v in summaries.items()])
    return _run("""You are a research assistant helping a graduate student identify gaps in the literature.
Based on the following paper summaries, identify:
1. **Questions left unanswered**
2. **Methodological limitations**
3. **Underrepresented contexts**
4. **Contradictions needing resolution**
5. **Future research opportunities**
Be specific and reference individual papers when relevant.
Paper summaries: {text}
Research Gaps Analysis:""", text=combined)


def compare_papers(question: str, full_texts: dict) -> str:
    combined = "\n\n".join([f"=== {os.path.basename(k)} ===\n{v[:2000]}" for k, v in full_texts.items()])
    return _run("""You are a research assistant specializing in comparative analysis of academic papers.
Compare the following papers on this specific question: {question}
For each paper note what it says (or doesn't say) about the topic.
Then provide a synthesis highlighting similarities, differences, and contradictions.
Papers: {text}
Comparative Analysis:""", question=question, text=combined)


def ask_primary_paper(question: str, primary_text: str, primary_name: str, other_texts: dict) -> str:
    """Answer a question using one paper as the primary source, comparing others against it."""
    others = "\n\n".join([f"=== {os.path.basename(k)} ===\n{v[:1500]}" for k, v in other_texts.items()])
    return _run("""You are a research assistant. The user has designated one paper as their PRIMARY source.
Answer the following question primarily from the PRIMARY paper, then note how the other papers
support, contradict, or extend what the primary paper says.

PRIMARY PAPER ({primary_name}):
{primary_text}

OTHER PAPERS:
{others}

Question: {question}

Answer (lead with what the primary paper says, then bring in the others):""",
        question=question,
        primary_name=os.path.basename(primary_name),
        primary_text=primary_text[:3000],
        others=others)


def suggest_questions(full_texts: dict) -> list:
    """Generate 10 good questions to ask about the ingested papers."""
    combined = "\n\n".join([f"Paper: {os.path.basename(k)}\n{v[:1500]}" for k, v in full_texts.items()])
    result = _run("""You are a research assistant helping a graduate student get the most out of their papers.
Based on the following paper excerpts, generate exactly 10 insightful questions the student should ask.
Include a mix of:
- Factual retrieval questions (specific findings, methods, definitions)
- Comparison questions (if multiple papers)
- Critical thinking questions (limitations, implications, gaps)

Return ONLY the 10 questions as a numbered list, nothing else.

Paper excerpts: {text}

10 Questions:""", text=combined)

    # Parse numbered list into a Python list
    lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
    questions = []
    for line in lines:
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:10]


def improve_writing(draft_text: str, context_texts: dict) -> str:
    """Improve a draft paragraph using ingested papers as sources."""
    context = "\n\n".join([f"From {os.path.basename(k)}:\n{v[:1500]}" for k, v in context_texts.items()])
    return _run("""You are an academic writing assistant helping a graduate student improve their writing.

The student has written the following draft passage. Using the research papers provided as context,
improve the passage by:
1. Strengthening claims with specific evidence from the papers
2. Improving clarity and academic tone
3. Adding citations where appropriate (use filename as reference)
4. Fixing any logical gaps or weak arguments

STUDENT DRAFT:
{draft}

AVAILABLE RESEARCH CONTEXT:
{context}

Improved passage (with inline citations to paper filenames):""",
        draft=draft_text, context=context)


def format_citations(source_docs: list, style: str = "APA") -> str:
    """Format retrieved source documents as properly formatted citations."""
    sources = []
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "")
        sources.append(f"File: {os.path.basename(source)}, Page: {page}\nExcerpt: {doc.page_content[:200]}")

    combined = "\n\n".join(sources)
    return _run("""You are a citation formatter. Based on the following source information extracted from PDF files,
generate properly formatted {style} citations. Since we only have filename and page information
(not full bibliographic data), create the best possible citation and note what information
would be needed to complete it properly.

Sources:
{text}

{style} Citations:""", style=style, text=combined)


def find_similar_paper(passage: str, summaries: dict) -> str:
    """Find which ingested paper is most similar to a given passage."""
    combined = "\n\n".join([f"Paper: {os.path.basename(k)}\nSummary: {v}" for k, v in summaries.items()])
    return _run("""You are a research assistant helping identify the most relevant paper for a given passage.

Given the following passage and paper summaries, identify:
1. Which paper is most thematically similar to the passage and why
2. How closely the passage aligns with each paper's core topic
3. Whether the passage could be citing or referencing any of these papers

PASSAGE:
{passage}

PAPER SUMMARIES:
{text}

Similarity Analysis:""", passage=passage, text=combined)


def analyze_image(image_bytes: bytes, question: str) -> str:
    client    = Groq()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response  = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": question or "Describe this figure from an academic paper in plain English. Explain what the data shows and what conclusions can be drawn."}
            ]
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content
