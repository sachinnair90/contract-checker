# Contract Checker

AI-powered contract integrity verification using Langchain, computer vision, and docum### Core Components

- **ContractScanner**: Uses docling to extract text, tables, and metadata from PDFs
- **SignatureDetector**: Uses YOLOv8 or DETR models to detect signatures in document images
- **Langchain Agent**: GPT-4 powered agent with tool-calling for flexible contract analysis
- **Langchain Chain**: Deterministic LCEL-based chain for sequential contract processing

### Tools

- `ContractScannerTool`: Extracts contract content and structure information
- `SignatureDetectorTool`: Identifies signatures and their locations in documentssing.

## Features

- **Contract Content Analysis**: Extract and analyze text content from PDF contracts using advanced document processing
- **Signature Detection**: Automatically detect signatures in contract documents using YOLO-based computer vision
- **AI-Powered Analysis**: Use GPT-4 to analyze contract completeness and validity
- **Modular Architecture**: Clean separation of scanning, detection, and analysis components

## Setup

### Prerequisites

- Python 3.13+
- UV package manager
- OpenAI API key
- HuggingFace token (for model downloads)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd contract-checker
```

2. Install dependencies:
```bash
uv sync
```

3. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Environment Variables

- `OPENAI_API_KEY`: Required for GPT-4 analysis
- `HF_TOKEN`: Required for downloading the signature detection model from HuggingFace

## Usage

### Basic Usage

Run the contract checker to test both approaches and models:

```bash
uv run python main.py
```

This will run analysis using:
- Agent + YOLO
- Agent + DETR  
- Chain + YOLO
- Chain + DETR

### Programmatic Usage

```python
from main import check_contract

# Agent-based approach (default)
result = check_contract("path/to/contract.pdf", model_type="yolo", use_chain=False)

# Chain-based approach
result = check_contract("path/to/contract.pdf", model_type="detr", use_chain=True)
```

### Architecture Options

The application provides two LangChain implementation approaches:

#### Agent-Based Approach (`use_chain=False`)
- Uses `create_openai_tools_agent` with tool-calling capabilities
- More flexible and conversational
- Can dynamically decide which tools to use and when
- Better for complex, multi-step reasoning

#### Chain-Based Approach (`use_chain=True`)
- Uses LCEL (LangChain Expression Language) for sequential processing
- Deterministic execution flow: scan → detect → analyze
- More predictable and easier to debug
- Better performance for straightforward workflows

### Comparison: Agent vs Chain

| Aspect | Agent-Based | Chain-Based |
|--------|-------------|-------------|
| Flexibility | High - can dynamically choose tools | Low - fixed execution order |
| Reasoning | Complex multi-step reasoning | Sequential processing |
| Performance | Variable (may call tools multiple times) | Consistent (single pass) |
| Debugging | Complex (agent decisions) | Simple (linear flow) |
| Use Case | Complex analysis requiring decisions | Standard contract checking |

### Custom Contract Analysis

Modify the `contract_path` variable in `main.py` to analyze different contracts:

```python
contract_path = "data/contracts/YourContract.pdf"
```

You can also specify which signature detection model to use:

```python
# Use YOLO model (default, faster)
analysis = check_contract(contract_path, model_type="yolo")

# Use DETR model (potentially more accurate)
analysis = check_contract(contract_path, model_type="detr")
```

## Architecture

### Core Components

- **ContractScanner**: Uses docling to extract text content and metadata from PDFs
- **SignatureDetector**: Uses YOLOv8 model to detect signatures in document images
- **Langchain Agent**: Orchestrates the analysis using GPT-4 with custom tools

### Tools

- `ContractScannerTool`: Extracts contract content and structure information
- `SignatureDetectorTool`: Identifies signatures and their locations in documents

## Development

### Project Structure

```
contract-checker/
├── main.py                 # Main application entry point
├── src/
│   └── utils/
│       ├── __init__.py
│       ├── contract_scanner.py    # PDF content extraction
│       └── signature_detector.py  # Signature detection
├── data/
│   └── contracts/          # Sample contract files
├── pyproject.toml          # Project dependencies
└── README.md
```

### Adding New Analysis Features

1. Create a new utility class in `src/utils/`
2. Implement a corresponding Langchain tool
3. Add the tool to the agent in `main.py`

## Dependencies

- **docling**: Advanced document processing and layout analysis
- **ultralytics**: YOLOv8 for signature detection
- **langchain**: AI agent framework
- **transformers**: For DETR model support
- **supervision**: Computer vision utilities
- **PyMuPDF**: PDF processing

## Model Information

- **Signature Detection (YOLO)**: Uses `tech4humans/yolov8s-signature-detector` from HuggingFace - fast and efficient
- **Signature Detection (DETR)**: Uses `tech4humans/conditional-detr-50-signature-detector` from HuggingFace - potentially more accurate with higher resolution processing
- **LLM**: GPT-4 via OpenAI API
- **Document Processing**: docling with table structure recognition

## License

See LICENSE file for details.