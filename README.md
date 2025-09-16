# LLM Chat Evaluation Suite

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-green.svg)](https://github.com/explodinggradients/ragas)

> Production-ready RAG evaluation framework with RAGAS metrics for financial domain applications.

**Note**: This project uses "Unity Financial Group" as a simulated company for demonstration purposes.

## Overview

A comprehensive RAG system evaluation framework that combines LangChain's RAG capabilities with RAGAS (Retrieval-Augmented Generation Assessment) metrics. Designed for financial domain applications with industry-standard quality benchmarks.

**Key Features:**
- 5 core RAGAS metrics (Answer Relevancy, Context Precision, Context Recall, Faithfulness, Answer Correctness)
- 50 financial domain questions across 10 categories
- Banking/finance industry thresholds
- Real-time performance monitoring
- A/B testing support

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/llm-chat-eval-suite.git
cd llm-chat-eval-suite
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run RAGAS evaluation (Main feature)
python ragas_evaluation.py

# Start Streamlit chat interface (Testing)
streamlit run chat.py
```

## Architecture

```
User Query → RAG System → Vector Search → LLM Generation → RAGAS Evaluation → Metrics
```

**Core Components:**
- `ragas_evaluation.py`: **RAGAS metrics calculation** (Main evaluation system)
- `llm.py`: RAG system implementation (LangChain + OpenAI + Pinecone)
- `chat.py`: Streamlit chat interface (User testing)
- `config.py`: System configuration

## RAGAS Metrics

| Metric | Description | Financial Threshold |
|--------|-------------|-------------------|
| **Answer Relevancy** | Question-answer relevance | ≥ 0.70 (Good) |
| **Context Precision** | Retrieved context accuracy | ≥ 0.70 (Good) |
| **Context Recall** | Ground truth coverage | ≥ 0.70 (Good) |
| **Faithfulness** | Answer grounding in context | ≥ 0.75 (Good) |
| **Answer Correctness** | Accuracy vs ground truth | ≥ 0.70 (Good) |

## Usage

### RAGAS Evaluation (Main Feature)
```python
from ragas_evaluation import run_ragas_evaluation

# Run comprehensive evaluation with 53 financial questions
results = run_ragas_evaluation()
print(f"Answer Relevancy: {results['average_metrics']['answer_relevancy']:.3f}")
```

![RAGAS Evaluation Running](ragas_evaluation_running.png)

### Custom Question Evaluation
```python
from ragas_evaluation import evaluate_single_question

# Evaluate individual questions
result = evaluate_single_question(
    question="How do you calculate ROE?",
    ground_truth="ROE = Net Income / Shareholders' Equity"
)
```

### Streamlit Chat Interface (Testing)
```python
from llm import get_rag_chain

# For interactive testing
rag_chain = get_rag_chain()
answer = rag_chain.invoke({"input": "What is the current ratio?"})
```

## Technology Stack

**Core:**
- Python 3.11+, LangChain 0.3.3, OpenAI 1.51.2
- Pinecone 5.0.1 (vector DB), Chroma 0.5.13 (alternative)
- RAGAS (custom implementation)

**Data Processing:**
- python-docx 1.1.2, docx2txt 0.8, pypdf 4.3.1
- tiktoken 0.8.0 (tokenization)

**Web/API:**
- FastAPI 0.115.2, Uvicorn 0.31.1, HTTPX 0.27.2

## Project Structure

```
llm-chat-eval-suite/
├── ragas_evaluation.py          # 🎯 MAIN: RAGAS evaluation system
├── llm.py                       # RAG system implementation
├── chat.py                      # Streamlit chat interface (testing)
├── config.py                    # Configuration
├── create_guide_index.py        # Document indexing
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── ragas_evaluation_results.json # 📊 Evaluation results
```

## Configuration

```python
# config.py
class Config:
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 800
    search_k: int = 5
    score_threshold: float = 0.3
```

## Financial Domain Dataset

50 specialized questions across 10 categories:
- **Liquidity Ratios** (5): Current ratio, Quick ratio, Cash ratio
- **Profitability Ratios** (7): Net profit margin, ROE, ROA, ROI
- **Leverage Ratios** (5): Debt-to-equity, Interest coverage
- **Efficiency Ratios** (5): Inventory turnover, Asset turnover
- **Market Risk** (6): VIX, Beta, VaR, Sharpe ratio
- **Valuation** (5): P/E, P/B, EV/EBITDA, DCF
- **Cash Flow** (5): Free cash flow, Operating cash flow
- **Financial Statements** (5): Revenue vs Income, EPS
- **Risk Management** (5): Credit risk, Liquidity risk
- **Investment Analysis** (5): NPV, IRR, Payback period

## Performance Optimization

**Immediate (1 week):**
- Adjust search threshold: 0.3-0.5
- Improve prompts: More specific instructions
- Add few-shot examples: 3-5 high-quality examples
- Optimize chunk size: 800 characters

**Medium-term (1 month):**
- Hybrid search: BM25 + semantic search
- Custom metrics: Domain-specific evaluation
- Caching system: Redis integration
- Batch processing: Parallel evaluation

## API Reference

```python
# Core classes
class RAGSystem:
    def get_rag_chain(self) -> RunnableWithMessageHistory
    def get_retriever(self) -> BaseRetriever

class RAGASEvaluator:
    def evaluate_question(self, question: str, ground_truth: str) -> dict
    def run_full_evaluation(self) -> dict
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Install dev dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest tests/`
5. Submit pull request

**Contribution Areas:**
- New evaluation metrics
- Additional LLM support
- Performance optimization
- Documentation improvements


## References

- **RAGAS Paper**: Es, S., et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv:2309.15217*
- **LangChain Docs**: https://python.langchain.com/
- **RAGAS GitHub**: https://github.com/explodinggradients/ragas
- **NIST AI RMF**: https://www.nist.gov/itl/ai-risk-management-framework
- **AI EU Act**: https://eur-lex.europa.eu/eli/reg/2024/1689/oj

---

