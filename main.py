"""
Contract Checker - Langchain-based contract integrity verification.
"""

import os

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.utils import ContractScanner, SignatureDetector


class ContractScannerTool(BaseTool):
    """Langchain tool for scanning contract content."""

    name: str = "contract_scanner"
    description: str = "Scan a PDF contract to extract text " "content and metadata"

    def __init__(self):
        super().__init__()
        self.scanner = ContractScanner()

    def _run(self, pdf_path: str) -> str:
        """Run the contract scanner on a PDF file."""
        try:
            result = self.scanner.scan_contract(pdf_path)
            return f"""Contract Scan Results:
- Filename: {result['filename']}
- Pages: {result['pages']}
- Tables: {result['tables']}
- Has Content: {result['has_content']}
- Content Preview: {result['content'][:500]}..."""
        except Exception as e:
            return f"Error scanning contract: {str(e)}"


class SignatureDetectorTool(BaseTool):
    """Langchain tool for detecting signatures in contracts."""

    name: str = "signature_detector"
    description: str = "Detect signatures in a PDF contract document"

    def __init__(self, model_type: str = "yolo"):
        super().__init__()
        self.detector = SignatureDetector(model_type=model_type)

    def _run(self, pdf_path: str) -> str:
        """Run signature detection on a PDF file."""
        try:
            result = self.detector.detect_signatures(pdf_path)
            return f"""Signature Detection Results:
- Filename: {result['filename']}
- Total Signatures: {result['total_signatures']}
- Pages with Signatures: {result['pages_with_signatures']}
- Has Signatures: {result['has_signatures']}"""
        except Exception as e:
            return f"Error detecting signatures: {str(e)}"


def create_contract_checker_agent(model_type: str = "yolo"):
    """Create a Langchain agent for contract checking."""
    # Load environment variables
    load_dotenv()

    # Check for required API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Initialize tools
    tools = [ContractScannerTool(), SignatureDetectorTool(model_type=model_type)]

    # Create LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Contract Integrity Checker AI. Your role is
to analyze contracts for completeness and validity.

You have access to tools that can:
1. Scan contract content to extract text and metadata
2. Detect signatures in the document

When analyzing a contract, you should:
- First scan the contract content
- Then check for signatures
- Analyze if the contract appears complete and properly signed
- Provide a summary of your findings

Always use the tools to gather information before making conclusions.""",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def check_contract(pdf_path: str, model_type: str = "yolo") -> str:
    """
    Check a contract for integrity using AI analysis.

    Args:
        pdf_path: Path to the PDF contract file
        model_type: Signature detection model to use ('yolo' or 'detr')

    Returns:
        Analysis results as a string
    """
    if not os.path.exists(pdf_path):
        return f"Error: Contract file not found: {pdf_path}"

    agent = create_contract_checker_agent(model_type=model_type)

    prompt = f"""
Please analyze the contract at path: {pdf_path}

Perform the following analysis:
1. Scan the contract content to understand what it contains
2. Check for signatures in the document
3. Determine if the contract appears complete and properly executed
4. Provide a summary of your findings

Be thorough in your analysis and explain your reasoning.
"""

    try:
        result = agent.invoke({"input": prompt})
        return result["output"]
    except Exception as e:
        return f"Error during contract analysis: {str(e)}"


def main():
    """Main function for the contract checker application."""
    print("Contract Checker - AI-powered contract integrity verification")
    print("=" * 60)

    # Example usage - you can modify this to accept command line arguments
    contract_path = "data/contracts/Contract291936Van_Gobbel.pdf"

    if os.path.exists(contract_path):
        # Test both YOLO and DETR models
        models_to_test = ["yolo", "detr"]

        for model_type in models_to_test:
            print(f"\n{'='*20} Testing with {model_type.upper()} " f"model {'='*20}")
            print(f"Analyzing contract: {contract_path}")
            print("-" * 40)

            analysis = check_contract(contract_path, model_type=model_type)
            print("\nAnalysis Results:")
            print(analysis)
            print("\n" + "=" * 60)
    else:
        print(f"Contract file not found: {contract_path}")
        print("Please ensure the contract file exists or update the path")


if __name__ == "__main__":
    main()
