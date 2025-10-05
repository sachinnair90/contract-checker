"""
Contract Checker - Langchain-based contract integrity verification.
"""

import os

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import ConfigDict

from src.utils import ContractScanner, SignatureDetector


class ContractScannerTool(BaseTool):
    """Langchain tool for scanning contract content."""

    name: str = "contract_scanner"
    description: str = "Scan a PDF contract to extract text " "content and metadata"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

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
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    if not azure_api_key or not azure_endpoint:
        raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required")

    # Initialize tools
    tools = [ContractScannerTool(), SignatureDetectorTool(model_type=model_type)]

    # Create LLM
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        api_key=azure_api_key,
        temperature=1,
    )

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


def create_contract_checker_chain(model_type: str = "yolo"):
    """Create a Langchain chain for contract checking
    (alternative to agent)."""
    # Load environment variables
    load_dotenv()

    # Check for required API keys
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    if not azure_api_key or not azure_endpoint:
        raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required")

    # Initialize utilities
    scanner = ContractScanner()
    detector = SignatureDetector(model_type=model_type)

    # Create LLM
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        api_key=azure_api_key,
        temperature=1,
    )

    # Create analysis prompt
    analysis_prompt = ChatPromptTemplate.from_template(
        """
You are a Contract Integrity Checker AI. Analyze the following contract
information and provide a comprehensive assessment.

Contract Scan Results:
{scan_results}

Signature Detection Results:
{signature_results}

Based on this information, please provide:
1. A summary of the contract content
2. Analysis of signature presence and validity
3. Assessment of contract completeness
4. Any potential issues or concerns
5. Overall recommendation

Be thorough and explain your reasoning.
"""
    )

    def scan_contract_step(inputs):
        """Step to scan contract content."""
        pdf_path = inputs["pdf_path"]
        try:
            result = scanner.scan_contract(pdf_path)
            return {
                "pdf_path": pdf_path,
                "scan_results": f"""Contract Scan Results:
- Filename: {result['filename']}
- Pages: {result['pages']}
- Tables: {result['tables']}
- Has Content: {result['has_content']}
- Content Preview: {result['content'][:1000]}...""",
                "model_type": model_type,
            }
        except Exception as e:
            return {
                "pdf_path": pdf_path,
                "scan_results": f"Error scanning contract: {str(e)}",
                "model_type": model_type,
            }

    def detect_signatures_step(inputs):
        """Step to detect signatures."""
        pdf_path = inputs["pdf_path"]
        try:
            result = detector.detect_signatures(pdf_path)
            return {
                **inputs,
                "signature_results": (
                    f"Signature Detection Results "
                    f"(using {result.get('model', model_type).upper()}): "
                    f"Filename: {result['filename']}, "
                    f"Total: {result['total_signatures']}, "
                    f"Pages: {result['pages_with_signatures']}, "
                    f"Has signatures: {result['has_signatures']}"
                ),
            }
        except Exception as e:
            return {
                **inputs,
                "signature_results": f"Error detecting signatures: {str(e)}",
            }

    # Create the chain using LCEL
    chain = (
        RunnableLambda(scan_contract_step)
        | RunnableLambda(detect_signatures_step)
        | analysis_prompt
        | llm
        | StrOutputParser()
    )

    return chain


def check_contract_with_agent(pdf_path: str, model_type: str = "yolo") -> str:
    """
    Check a contract for integrity using AI analysis (agent-based approach).

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


def check_contract_with_chain(pdf_path: str, model_type: str = "yolo") -> str:
    """
    Check a contract for integrity using AI analysis (chain-based approach).

    Args:
        pdf_path: Path to the PDF contract file
        model_type: Signature detection model to use ('yolo' or 'detr')

    Returns:
        Analysis results as a string
    """
    if not os.path.exists(pdf_path):
        return f"Error: Contract file not found: {pdf_path}"

    chain = create_contract_checker_chain(model_type=model_type)

    try:
        result = chain.invoke({"pdf_path": pdf_path})
        return result
    except Exception as e:
        return f"Error during contract analysis: {str(e)}"


# Alias for backward compatibility
def check_contract(
    pdf_path: str, model_type: str = "yolo", use_chain: bool = False
) -> str:
    """
    Check a contract for integrity using AI analysis.

    Args:
        pdf_path: Path to the PDF contract file
        model_type: Signature detection model to use ('yolo' or 'detr')
        use_chain: Whether to use chain-based approach (True) or
            agent-based (False)

    Returns:
        Analysis results as a string
    """
    if use_chain:
        return check_contract_with_chain(pdf_path, model_type)
    else:
        return check_contract_with_agent(pdf_path, model_type)


def main():
    """Main function for the contract checker application."""
    print("Contract Checker - AI-powered contract integrity verification")
    print("=" * 60)

    # Example usage - you can modify this to accept command line arguments
    contract_path = "data/contracts/Contract291936Van_Gobbel.pdf"

    if os.path.exists(contract_path):
        # Test both approaches and both models
        approaches = [("Agent", False), ("Chain", True)]
        models_to_test = ["yolo", "detr"]

        for approach_name, use_chain in approaches:
            for model_type in models_to_test:
                print(f"\n{'='*15} {approach_name} + {model_type.upper()} " f"{'='*15}")
                print(f"Analyzing contract: {contract_path}")
                print("-" * 40)

                analysis = check_contract(
                    contract_path, model_type=model_type, use_chain=use_chain
                )
                print("\nAnalysis Results:")
                print(analysis)
                print("\n" + "=" * 60)
    else:
        print(f"Contract file not found: {contract_path}")
        print("Please ensure the contract file exists or update the path")


if __name__ == "__main__":
    main()
