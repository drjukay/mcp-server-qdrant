import os
from typing import Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "OpenAI package not found. Install it with: pip install openai"
    )

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    :param model_name: The name of the OpenAI embedding model to use.
    :param api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it as a parameter or set OPENAI_API_KEY environment variable."
            )
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Model-specific dimensions mapping
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        # Validate model
        if model_name not in self._model_dimensions:
            raise ValueError(
                f"Unsupported OpenAI embedding model: {model_name}. "
                f"Supported models: {list(self._model_dimensions.keys())}"
            )

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        response = await self.client.embeddings.create(
            input=documents,
            model=self.model_name
        )
        
        return [embedding.embedding for embedding in response.data]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        response = await self.client.embeddings.create(
            input=[query],
            model=self.model_name
        )
        
        return response.data[0].embedding

    def get_vector_name(self) -> str:
        """Get the name of the vector for the Qdrant collection."""
        # Return empty string for single vector collections (legacy compatibility)
        # This makes it compatible with existing databases that use single vectors
        return ""

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self._model_dimensions[self.model_name]