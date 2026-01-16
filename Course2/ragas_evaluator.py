from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], openai_api_key: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not isinstance(question, str) or not question.strip():
        return {"error": "Invalid input: question must be a non-empty string."}
    if not isinstance(answer, str) or not answer.strip():
        return {"error": "Invalid input: answer must be a non-empty string."}
    if not isinstance(contexts, list) or not contexts:
        return {"error": "Invalid input: contexts must be a non-empty list of strings."}
 

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # Get API key from parameter or environment variable
    import os
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key not provided"}
    
     # TODO: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
    )

    # TODO: Create evaluator_embeddings with model test-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    )

    # TODO: Define an instance for each metric to evaluate
    # Note: Only using metrics that don't require reference answers
    # RougeScore, BleuScore, and NonLLMContextPrecisionWithReference require reference data
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
    ]

    # TODO: Evaluate the response using the metrics
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    # Use the correct RAGAS API - newer versions use dataset parameter
    try:
        # Try with dataset parameter (newer RAGAS versions)
        from ragas import EvaluationDataset
        dataset = EvaluationDataset(samples=[sample])
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
    except (TypeError, ImportError):
        # Fallback to direct list (older RAGAS versions)
        try:
            results = evaluate(
                [sample],
                metrics=metrics,
            )
        except TypeError:
            # Last resort - try with samples kwarg
            results = evaluate(
                samples=[sample],
                metrics=metrics,
            )

    # TODO: Return the evaluation results
    return results.to_pandas().iloc[0].to_dict()
