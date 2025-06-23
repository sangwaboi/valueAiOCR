#!/usr/bin/env python3
"""
PDF Document Reconstruction System (OpenAI Only)
- Extracts and OCRs each page
- Uses OpenAI LLM for summarization and ordering
- Outputs a reordered PDF with Table of Contents
"""
import os
import re
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import click
import numpy as np
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
import easyocr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from tqdm import tqdm
import textwrap
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PageInfo:
    index: int
    text: str
    image_path: str
    summary: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class PDFReconstructor:
    def __init__(self, llm_provider: str = "openai", llm_key: str = None, model_name: str = None):
        self.llm_provider = llm_provider.lower()
        self.llm_key = llm_key
        # Set default model_name based on provider if not given
        if model_name is None:
            if self.llm_provider == "openai":
                model_name = "gpt-3.5-turbo"
            elif self.llm_provider == "gemini":
                model_name = "gemini-2.0-flash"
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        self.model_name = model_name
        self.ocr_reader = easyocr.Reader(['en'])
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pages: List[PageInfo] = []
        self.similarity_matrix = None
        self.ordered_pages = []
        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.1, openai_api_key=self.llm_key)
        elif self.llm_provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.1, google_api_key=self.llm_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def pdf_to_images(self, pdf_path: str, out_dir: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        images = convert_from_path(pdf_path)
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(out_dir, f"page_{i+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
        return image_paths

    def ocr_pages(self, image_paths: List[str]):
        for idx, img_path in enumerate(tqdm(image_paths, desc="OCR")):
            result = self.ocr_reader.readtext(img_path, detail=0, paragraph=True)
            text = "\n".join(result)
            self.pages.append(PageInfo(index=idx, text=text, image_path=img_path))

    def extract_metadata(self):
        for page in self.pages:
            meta = {}
            # Page number pattern
            match = re.search(r"Page\s*(\d+)\s*of\s*(\d+)", page.text, re.IGNORECASE)
            if match:
                meta['page_number'] = int(match.group(1))
                meta['total_pages'] = int(match.group(2))
            # Section header pattern
            header = re.search(r"(Application Form|KYC|Summary|Loan|Agreement|Statement|Schedule)", page.text, re.IGNORECASE)
            if header:
                meta['section'] = header.group(1)
            # Date pattern
            date = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", page.text)
            if date:
                meta['date'] = date.group(1)
            page.metadata = meta

    def embed_pages(self):
        texts = [p.text for p in self.pages]
        embeddings = self.sentence_model.encode(texts)
        for i, emb in enumerate(embeddings):
            self.pages[i].embedding = emb
        self.similarity_matrix = cosine_similarity(embeddings)

    def summarize_pages(self):
        batch_size = 14 if self.llm_provider == "gemini" else len(self.pages)
        wait_time = 60 if self.llm_provider == "gemini" else 0
        total_pages = len(self.pages)
        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size, total_pages)
            for i in tqdm(range(start, end), desc=f"LLM Summarize [{start+1}-{end}]"):
                page = self.pages[i]
                prompt = f"Summarize the following scanned document page for ordering and table of contents.\n\n{textwrap.shorten(page.text, width=2000)}"
                try:
                    resp = self.llm([HumanMessage(content=prompt)])
                    page.summary = resp.content.strip()
                except Exception as e:
                    logger.error(f"LLM error on page {page.index+1}: {e}")
                    page.summary = "(Summary unavailable)"
            # If Gemini, wait after each batch to respect quota
            if self.llm_provider == "gemini" and end < total_pages:
                logger.info(f"Gemini quota: processed {end} pages, waiting {wait_time} seconds to avoid rate limit...")
                time.sleep(wait_time)

    def order_pages(self):
        # Use LLM summaries to order pages semantically
        summaries = [p.summary for p in self.pages]
        prompt = (
            "Given the following page summaries from a jumbled loan/mortgage PDF, "
            "output the most probable correct order as a list of page indices (0-based, comma-separated).\n\n"
        )
        for i, s in enumerate(summaries):
            prompt += f"Page {i}: {s}\n"
        prompt += "\nOrder:"
        try:
            resp = self.llm([HumanMessage(content=prompt)])
            order = [int(x) for x in re.findall(r'\d+', resp.content)]
            self.ordered_pages = [self.pages[i] for i in order if 0 <= i < len(self.pages)]
        except Exception as e:
            logger.error(f"LLM ordering error: {e}")
            self.ordered_pages = self.pages

    def generate_toc(self) -> List[str]:
        toc = []
        for i, page in enumerate(self.ordered_pages):
            section = page.metadata.get('section', 'Page') if page.metadata else 'Page'
            toc.append(f"{i+1}. {section} (original page {page.index+1})")
        return toc

    def create_output_pdf(self, output_pdf: str, toc: List[str]):
        # Create ToC PDF
        toc_pdf = "toc_temp.pdf"
        doc = SimpleDocTemplate(toc_pdf, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Table of Contents", styles['Title']), Spacer(1, 12)]
        for entry in toc:
            story.append(Paragraph(entry, styles['Normal']))
        doc.build(story)
        # Merge ToC and ordered pages
        writer = PyPDF2.PdfWriter()
        with open(toc_pdf, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                writer.add_page(page)
        for page in self.ordered_pages:
            img = Image.open(page.image_path).convert("RGB")
            img_pdf = page.image_path + ".pdf"
            img.save(img_pdf, "PDF")
            with open(img_pdf, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    writer.add_page(p)
            os.remove(img_pdf)
        with open(output_pdf, "wb") as f:
            writer.write(f)
        os.remove(toc_pdf)

@click.command()
@click.argument('input_pdf', type=click.Path(exists=True))
@click.argument('output_pdf', type=click.Path())
@click.option('--llm-provider', default='openai', help='LLM provider: openai or gemini')
@click.option('--llm-key', required=True, help='API key for the selected LLM provider')
@click.option('--llm-model', default=None, help='Model name for the selected LLM provider (default: gpt-3.5-turbo for OpenAI, gemini-2.0-flash for Gemini)')
@click.option('--tmp-dir', default='tmp_pages', help='Temporary directory for images')
def main(input_pdf, output_pdf, llm_provider, llm_key, llm_model, tmp_dir):
    recon = PDFReconstructor(llm_provider=llm_provider, llm_key=llm_key, model_name=llm_model)
    logger.info(f"Converting PDF to images...")
    image_paths = recon.pdf_to_images(input_pdf, tmp_dir)
    logger.info(f"Running OCR on {len(image_paths)} pages...")
    recon.ocr_pages(image_paths)
    logger.info("Extracting metadata...")
    recon.extract_metadata()
    logger.info("Embedding pages...")
    recon.embed_pages()
    logger.info("Summarizing pages with LLM...")
    recon.summarize_pages()
    logger.info("Ordering pages with LLM...")
    recon.order_pages()
    logger.info("Generating Table of Contents...")
    toc = recon.generate_toc()
    logger.info("Creating output PDF...")
    recon.create_output_pdf(output_pdf, toc)
    logger.info(f"Done! Output: {output_pdf}")

if __name__ == "__main__":
    main() 