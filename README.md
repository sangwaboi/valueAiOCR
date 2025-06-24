# valueAiOCR

A web-based tool to reconstruct, reorder, and summarize jumbled scanned PDFs (like loan/mortgage files) using OCR, semantic analysis, and LLMs.

---

## 🚀 Project Overview

**valueAiOCR** lets you upload a messy, scanned PDF and get back an intelligently ordered, summarized, and table-of-contents-enhanced PDF. It uses OCR to extract text, AI to understand and order pages, and a modern web interface for ease of use.

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

---

## 🌐 Deployment

- **Backend:** FastAPI, hosted on Render ([https://valueaiocr.onrender.com](https://valueaiocr.onrender.com))
- **Frontend:** React, hosted on Vercel ([https://value-ai-ocr.vercel.app](https://value-ai-ocr.vercel.app))

---

## 🎥 Video Walkthrough

Watch a full demo and explanation here:  
[https://www.loom.com/share/5e25aa1e37884504b15035d48bc47ee1?sid=9cecb391-37ec-40f3-8d46-a2acbc5ccd19](https://www.loom.com/share/5e25aa1e37884504b15035d48bc47ee1?sid=9cecb391-37ec-40f3-8d46-a2acbc5ccd19)

---

## 📂 Adding Sample Input/Output PDFs

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