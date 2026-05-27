"""
analyzer.py — Direct LLM analysis using Groq (no retrieval)
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL = "llama3-8b-8192"

PROMPTS = {
    "Title": """You are a research assistant helping a graduate student understand academic papers.

Analyze the following paper title and explain in plain English:
1. What topic or problem the paper is about
2. Who the likely subjects or context are (if mentioned)
3. What kind of study this appears to be (experiment, review, analysis, etc.)
4. Any important keywords or technical terms and what they mean

Title: {text}

Give a clear, friendly explanation a first-year grad student would understand.""",

    "Abstract": """You are a research assistant helping a graduate student understand academic papers.

Break down the following abstract in plain English. Cover:
1. What problem or question the study addresses
2. What methods were used
3. What the main findings were
4. What the authors concluded
5. Why this research matters

Abstract: {text}

Be clear and specific. Avoid jargon where possible, and explain any technical terms you do use.""",

    "Introduction": """You are a research assistant helping a graduate student understand academic papers.

Analyze the following introduction and explain:
1. What background context is being established
2. What gap in knowledge the authors identified
3. What the research question or hypothesis is
4. How this study fits into the existing literature

Introduction: {text}

Explain in plain English suitable for a graduate student new to the topic.""",

    "Methods": """You are a research assistant helping a graduate student understand academic papers.

Break down the following methods section and explain:
1. What type of study design was used
2. Who or what the subjects/samples were
3. What procedures or techniques were applied
4. What was measured and how
5. Any potential weaknesses or limitations in the approach

Methods: {text}

Be specific and plain. Flag anything that seems unusual or worth questioning.""",

    "Results": """You are a research assistant helping a graduate student understand academic papers.

Analyze the following results section and explain:
1. What the main findings were in plain language
2. Whether the results supported or contradicted the hypothesis
3. Any surprising or noteworthy patterns in the data
4. What the numbers or statistics actually mean in practical terms

Results: {text}

Translate any statistical language into plain English.""",

    "Discussion / Conclusion": """You are a research assistant helping a graduate student understand academic papers.

Analyze the following discussion or conclusion and explain:
1. How the authors interpreted their findings
2. Whether you think the interpretations are well-supported
3. What limitations the authors acknowledged
4. What future research directions are suggested
5. What the real-world implications are

Discussion/Conclusion: {text}

Be critical but fair. Note if any claims seem overstated.""",

    "Custom section": """You are a research assistant helping a graduate student understand academic papers.

Analyze the following text from an academic paper and provide:
1. A plain-English summary of what it is saying
2. An explanation of any technical terms or jargon
3. The key takeaway or main point
4. Any questions a critical reader should ask about this section

Text: {text}

Be clear, thorough, and accessible.""",
}


def analyze_text(text: str, section_type: str) -> str:
    prompt_template = PROMPTS.get(section_type, PROMPTS["Custom section"])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm    = ChatGroq(model=LLM_MODEL)
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def summarize_paper(full_text: str, filename: str) -> str:
    excerpt = full_text[:4000]

    prompt_template = """You are a research assistant helping a graduate student quickly understand papers.

Based on the following excerpt from a research paper, write a brief plain-English abstract (4-6 sentences) that covers:
1. What problem or question the paper addresses
2. What approach or method was used
3. What the main findings or contributions are
4. Why it matters or who it's relevant to

Be concise, clear, and avoid jargon where possible. Write it as a paragraph, not a list.

Paper excerpt:
{text}

Plain-English Abstract:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm    = ChatGroq(model=LLM_MODEL)
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"text": excerpt})
