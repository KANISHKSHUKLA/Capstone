# Research Paper Summarizer

Ever found yourself drowning in complex research papers? Me too! That's why I built this tool - a friendly, AI-powered assistant that helps you digest academic papers without the headache. It extracts the important stuff and lets you have an actual conversation about the paper's content.

## What it does

### For readers & researchers
- **Upload any research paper** (PDF) and get instant insights
- **Chat with your paper** - ask questions about methods, results, or anything unclear
- **Get the complete picture** through multiple analysis angles:
  - Main concepts and technologies (the "what")
  - Problem and alternative approaches (the "why")
  - Methodology breakdown (the "how")
  - Implementation ideas with pseudo-code (the "build it")

### For developers
- **Modern stack**: React/Next.js frontend with FastAPI backend
- **Fast processing**: Parallel AI analysis using Groq's deepseek models
- **Clean architecture**: Separation of concerns between UI, API, and AI processing

## Getting Started

### Backend Setup
1. Clone this repo
2. Install Python dependencies:
   ```bash
   cd research-paper-summarizer
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the `app` directory with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   You can get an API key by signing up at [groq.com](https://console.groq.com/keys)
4. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Next.js dev server:
   ```bash
   npm run dev
   ```
3. Navigate to http://localhost:3000 in your browser

## Key Features

### Paper Analysis
- **Smart extraction** of text from research PDFs
- **Multi-faceted analysis** running in parallel for speed
- **Structured results** for easy navigation and comprehension

### Interactive UI
- **Clean, responsive design** that works on all devices
- **Draggable chat panel** that you can resize to your needs
- **Code syntax highlighting** for implementation snippets

### Chat Experience
- **Contextual conversations** about the paper's content
- **Intelligent responses** that reference specific sections
- **Visual loading indicators** while waiting for responses

## Architecture

This project combines a powerful backend with a sleek frontend:

### Backend (Python/FastAPI)
- **API Layer**: RESTful endpoints for paper processing and chat
- **Core Engine**: Coordinates PDF extraction and AI analysis
- **LLM Integration**: Context-aware prompting for the deepseek model

### Frontend (Next.js/React)
- **Modern UI**: Clean, minimalist interface with Tailwind CSS
- **Dynamic Routing**: Client-side navigation between views
- **Component Architecture**: Reusable, maintainable UI elements

## Project Structure

```
research-paper-summarizer/
├── app/                  # Backend FastAPI application
│   ├── api/             # API routes and endpoint handlers
│   ├── clients/         # LLM API clients (Groq)
│   ├── core/            # Business logic and chat management
│   ├── models/          # Data models and schemas
│   ├── prompts/         # LLM prompting templates
│   └── utils/           # Utility functions
├── frontend/            # Next.js frontend application
│   ├── app/             # Next.js app router pages
│   ├── components/      # Reusable React components
│   ├── styles/          # CSS and styling
│   └── lib/             # Frontend utility functions
└── README.md            # You are here!
```

## API Endpoints

- `POST /api/analyze` - Submit a paper for analysis
- `GET /api/jobs/{job_id}` - Get analysis results
- `POST /api/jobs/{job_id}/chat` - Send a message about a paper
- `GET /api/jobs/{job_id}/chat` - Get chat history

## Future Improvements

- Citation tracking and verification
- Multi-paper comparison
- Export of analysis in different formats
- Mobile app version

---

Built with ❤️ for researchers and curious minds. If you've got questions or ideas for improvement, feel free to open an issue or PR!
