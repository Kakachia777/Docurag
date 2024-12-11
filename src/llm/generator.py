from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
COMPLETION_COUNTER = Counter(
    "llm_completions_total",
    "Total number of LLM completion requests"
)
COMPLETION_LATENCY = Histogram(
    "llm_completion_latency_seconds",
    "Time spent on LLM completion requests"
)
TOKEN_COUNTER = Counter(
    "llm_tokens_total",
    "Total number of tokens processed",
    ["type"]  # prompt or completion
)

class ResponseGenerator:
    """Generates responses using LLM with context from retrieved documents."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the assistant."""
        return """You are a helpful technical documentation assistant. Your role is to:
1. Provide accurate and relevant information based on the documentation provided
2. Explain technical concepts clearly and concisely
3. Include relevant code examples when appropriate
4. Admit when you're not sure about something
5. Stick to the information in the provided documentation

Format your responses in Markdown when appropriate."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _generate_completion(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate completion with retry logic.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            OpenAI API response
        """
        try:
            with COMPLETION_LATENCY.time():
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            COMPLETION_COUNTER.inc()
            TOKEN_COUNTER.labels(type="prompt").inc(response["usage"]["prompt_tokens"])
            TOKEN_COUNTER.labels(type="completion").inc(response["usage"]["completion_tokens"])
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM with retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            chat_history: Optional chat history
            
        Returns:
            Generated response with metadata
        """
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[From: {chunk['metadata'].get('source', 'Unknown')}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history[-5:])  # Include last 5 messages
        
        # Add context and query
        messages.extend([
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ])
        
        # Generate response
        start_time = datetime.utcnow()
        response = self._generate_completion(messages)
        
        # Extract and format response
        generated_text = response["choices"][0]["message"]["content"]
        
        return {
            "response": generated_text,
            "metadata": {
                "model": self.model,
                "temperature": self.temperature,
                "tokens_used": response["usage"]["total_tokens"],
                "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                "sources": [chunk["metadata"].get("source") for chunk in retrieved_chunks]
            }
        }
    
    def generate_followup_questions(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate relevant follow-up questions based on the context.
        
        Args:
            query: Original user query
            response: Generated response
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            List of follow-up questions
        """
        prompt = f"""Based on the user's question and the provided response, suggest 3 relevant follow-up questions.
        
Original question: {query}

Response summary: {response[:200]}...

Context: {retrieved_chunks[0]['text'][:200]}...

Generate 3 concise follow-up questions that would help the user learn more about this topic."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._generate_completion(messages)
        questions = response["choices"][0]["message"]["content"].strip().split("\n")
        
        # Clean and format questions
        questions = [q.strip().strip("123.") for q in questions if q.strip()]
        return questions[:3]  # Ensure we return at most 3 questions 