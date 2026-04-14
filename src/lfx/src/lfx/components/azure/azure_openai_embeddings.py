import time

from langchain_openai import AzureOpenAIEmbeddings

from lfx.base.models.model import LCModelComponent
from lfx.base.models.openai_constants import OPENAI_EMBEDDING_MODEL_NAMES
from lfx.field_typing import Embeddings
from lfx.io import DropdownInput, FloatInput, IntInput, MessageTextInput, Output, SecretStrInput


class RateLimitedEmbeddings(Embeddings):
    def __init__(self, embeddings: Embeddings, delay: float, max_retries: int):
        self._embeddings = embeddings
        self._delay = delay
        self._max_retries = max_retries

    def _sleep_and_retry(self, e, retry_count):
        import re
        import time

        msg = str(e)
        match = re.search(r"retry after (\d+) seconds", msg, re.IGNORECASE)
        if match:
            wait_time = int(match.group(1)) + 1
        else:
            wait_time = (2**retry_count) * 10

        print(f"Rate limit hit: {msg}. Sleeping for {wait_time}s...")
        time.sleep(wait_time)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import time

        from openai import RateLimitError

        all_embeddings = []
        chunk_size = getattr(self._embeddings, "chunk_size", 100)
        for i in range(0, len(texts), chunk_size):
            if i > 0 and self._delay > 0:
                time.sleep(self._delay)

            chunk = texts[i : i + chunk_size]
            retries = 0
            while True:
                try:
                    all_embeddings.extend(self._embeddings.embed_documents(chunk))
                    break
                except RateLimitError as e:
                    if retries >= self._max_retries:
                        raise
                    self._sleep_and_retry(e, retries)
                    retries += 1
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        import re

        from openai import RateLimitError

        all_embeddings = []
        chunk_size = getattr(self._embeddings, "chunk_size", 100)
        for i in range(0, len(texts), chunk_size):
            if i > 0 and self._delay > 0:
                await asyncio.sleep(self._delay)

            chunk = texts[i : i + chunk_size]
            retries = 0
            while True:
                try:
                    results = await self._embeddings.aembed_documents(chunk)
                    all_embeddings.extend(results)
                    break
                except RateLimitError as e:
                    if retries >= self._max_retries:
                        raise
                    msg = str(e)
                    match = re.search(r"retry after (\d+) seconds", msg, re.IGNORECASE)
                    wait_time = int(match.group(1)) + 1 if match else (2**retries) * 10
                    print(f"Rate limit hit: {msg}. Sleeping for {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    retries += 1
        return all_embeddings

    async def aembed_query(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)

    def __getattr__(self, name):
        return getattr(self._embeddings, name)


class AzureOpenAIEmbeddingsComponent(LCModelComponent):
    display_name: str = "Azure OpenAI Embeddings"
    description: str = "Generate embeddings using Azure OpenAI models."
    documentation: str = "https://python.langchain.com/docs/integrations/text_embedding/azureopenai"
    icon = "Azure"
    name = "AzureOpenAIEmbeddings"

    API_VERSION_OPTIONS = [
        "2022-12-01",
        "2023-03-15-preview",
        "2023-05-15",
        "2023-06-01-preview",
        "2023-07-01-preview",
        "2023-08-01-preview",
    ]

    inputs = [
        DropdownInput(
            name="model",
            display_name="Model",
            advanced=False,
            options=OPENAI_EMBEDDING_MODEL_NAMES,
            value=OPENAI_EMBEDDING_MODEL_NAMES[0],
        ),
        MessageTextInput(
            name="azure_endpoint",
            display_name="Azure Endpoint",
            required=True,
            info="Your Azure endpoint, including the resource. Example: `https://example-resource.azure.openai.com/`",
        ),
        MessageTextInput(
            name="azure_deployment",
            display_name="Deployment Name",
            required=True,
        ),
        DropdownInput(
            name="api_version",
            display_name="API Version",
            options=API_VERSION_OPTIONS,
            value=API_VERSION_OPTIONS[-1],
            advanced=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="Azure OpenAI API Key",
            required=True,
        ),
        IntInput(
            name="dimensions",
            display_name="Dimensions",
            info="The number of dimensions the resulting output embeddings should have. "
            "Only supported by certain models.",
            advanced=True,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            advanced=True,
            value=100,
            info="The number of documents to embed in a single request. Smaller values help avoid rate limits.",
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            advanced=True,
            value=5,
            info="Maximum number of retries for the API request.",
        ),
        FloatInput(
            name="delay",
            display_name="Delay",
            advanced=True,
            value=1.0,
            info="Delay in seconds between batches of embeddings to avoid rate limits.",
        ),
    ]

    outputs = [
        Output(display_name="Embeddings", name="embeddings", method="build_embeddings"),
    ]

    def build_embeddings(self) -> Embeddings:
        try:
            embeddings = AzureOpenAIEmbeddings(
                model=self.model,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                api_version=self.api_version,
                api_key=self.api_key,
                dimensions=self.dimensions or None,
                chunk_size=self.chunk_size,
                max_retries=self.max_retries,
            )
        except Exception as e:
            msg = f"Could not connect to AzureOpenAIEmbeddings API: {e}"
            raise ValueError(msg) from e

        return RateLimitedEmbeddings(embeddings, self.delay, self.max_retries)
