# valueAiOCR

A solution for the Value AI Labs GenAI Internship Take-Home Assignment: **Reconstruct the Jumbled PDF**

---

## 📝 Problem Statement

> **Challenge:**
> You've been handed a multi-page scanned PDF where the pages are randomly shuffled — a surprisingly common issue in the mortgage and lending industry. Your task is to build a tool that can analyze and reorder the pages to restore the document to its likely original sequence.

---

## 📂 Sample Input/Output PDFs

- **Input (Jumbled) PDF:** [Download from Google Drive](https://drive.google.com/file/d/1-eSLcSWlnQiyHv2aX7ATuY8pjVGnzny9/view?usp=sharing)
- **Output (Ordered) PDF:** [Download from Google Drive](https://drive.google.com/file/d/1dlSJRUChBlMFQ6aHaRvt6U9ZhuaAdTfC/view?usp=sharing)

---

## 🎥 Video Walkthrough

Watch a full demo and explanation here:  
[https://www.loom.com/share/5e25aa1e37884504b15035d48bc47ee1?sid=9cecb391-37ec-40f3-8d46-a2acbc5ccd19](https://www.loom.com/share/5e25aa1e37884504b15035d48bc47ee1?sid=9cecb391-37ec-40f3-8d46-a2acbc5ccd19)

---


## 🛠️ Workflow

1. **Upload PDF** via the web interface (React frontend)
2. **PDF to Images:** Each page is converted to an image
3. **OCR:** Text is extracted from each image (Tesseract/EasyOCR)
4. **Semantic Analysis:** Each page is embedded using sentence-transformers
5. **LLM Summarization & Ordering:**
   - Google Gemini LLM summarizes and suggests the correct order for pages
   - Handles API rate limits by batching requests
6. **PDF Reconstruction:** Pages are reordered, a table of contents is generated, and a new PDF is created
7. **Download:** User downloads the organized PDF
8. **Logs & Reasoning:** The backend produces a JSON report explaining the ordering and summarization

---

## 🌐 Deployment

- **Backend:** FastAPI, hosted on Render ([https://valueaiocr.onrender.com](https://valueaiocr.onrender.com))
- **Frontend:** React, hosted on Vercel ([https://value-ai-ocr.vercel.app](https://value-ai-ocr.vercel.app))


### **What This App Does**
- Accepts a jumbled PDF as input
- Analyzes page content using OCR, LLMs, and embeddings
- Outputs a reordered PDF
- Provides logs, reasoning, and natural language notes explaining the page order
- **Bonus:**
  - Adds a Table of Contents based on detected sections
  - Identifies missing or duplicate pages (if any)

---

## 🚀 Project Overview

**valueAiOCR** lets you upload a messy, scanned PDF and get back an intelligently ordered, summarized, and table-of-contents-enhanced PDF. It uses OCR to extract text, AI to understand and order pages, and a modern web interface for ease of use.

---

## 📂 Adding Your Own PDFs

You can add your own sample PDFs and outputs to the repository for demo or testing purposes:

1. **Input PDFs:** Place in the `uploads/` directory (e.g., `uploads/sample_input.pdf`)
2. **Output PDFs:** Place in the `outputs/` directory (e.g., `outputs/sample_output.pdf`)
3. **Text/JSON outputs:** Place in the project root or a new `results/` directory

**To add and push:**
```sh
# Copy your files into the correct folders
cp /path/to/your/input.pdf uploads/
cp /path/to/your/output.pdf outputs/

# Stage, commit, and push
git add uploads/ outputs/
git commit -m "Add sample input and output PDFs"
git push
```

---

## 🧑‍💻 Contributing
- Fork the repo, make your changes, and submit a pull request!
- For large files, consider sharing via cloud storage and linking in the README.

---

## 💡 Future Improvements
- User authentication & job queue
- Real-time progress bar
- Multi-language OCR
- Advanced table of contents
- Analytics dashboard

---

**Questions?** Open an issue or reach out!
