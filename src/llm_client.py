import os
from typing import Optional
import asyncio
from asyncio import Semaphore
import httpx
import random
from dotenv import load_dotenv
from functools import lru_cache
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
CHAT_MODEL = os.getenv("CHAT_MODEL", "alibayram/medgemma:latest") #"alibayram/medgemma:27B"
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.5))
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:11434/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text") #"bge-m3:latest"
headers = {"Content-Type": "application/json"}
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"


def make_payload(model: str, prompt: str, temperature: float) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }

@lru_cache(maxsize=2048)
def _cached_embedding_request(text: str) -> list[float] | None:
    request = make_payload(EMBEDDING_MODEL, text, TEMPERATURE)
    try:
        timeout = 60.0
        with httpx.Client(timeout=timeout) as client:
            response = client.post(EMBEDDING_URL, json=request, headers=headers)
            response.raise_for_status()
            return response.json().get("embedding")
    except httpx.HTTPError as e:
        logger.error(f"Cached embedding request failed: {e}")
        return None



class LLMClient:
    def __init__(
            self,
            embedding_model: str = EMBEDDING_MODEL,
            chat_model: str = CHAT_MODEL,
            chat_temperature: float = TEMPERATURE
    ):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.chat_temperature = chat_temperature

    def embedding_request(self, text: str):
        """ Create an API embedding request for input text. """
        request_content = make_payload(self.embedding_model, text, self.chat_temperature)
        return request_content

    def chat_request(self, prompt: str):
        """ Create an API chat request from system and user prompts. """
        request_content = make_payload(self.chat_model, prompt, self.chat_temperature)
        return request_content


class HEALpacaClient(LLMClient):
    def __init__(
            self,
            chat_model: str = CHAT_MODEL,
            embedding_model: str = EMBEDDING_MODEL,
            api_url: str = LLM_API_URL,
            embedding_url: str = EMBEDDING_URL,
            chat_temperature: float = TEMPERATURE,
    ):
        super().__init__(chat_model=chat_model, embedding_model=embedding_model, chat_temperature=chat_temperature)
        self.api_url = api_url
        self.embedding_url = embedding_url

    def get_embedding(self, text: str) -> list[float] | None:
        return _cached_embedding_request(text)

    def get_chat_completion(self, prompt: str) -> str | None:
        """ Get single chat response with improved timeout """
        request = self.chat_request(prompt)
        try:
            # Increased timeout for unstable server
            with httpx.Client(timeout=300.0) as client:  # 5 minutes instead of 2
                response = client.post(self.api_url, json=request, headers=headers)
                response.raise_for_status()
                return response.json().get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Chat Completion request failed: {e}")
            return None

    def get_embeddings(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Get embeddings for a list of texts (synchronously)."""
        results = []
        for text in texts:
            try:
                result = self.get_embedding(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Embedding failed for text: {text}. Error: {e}")
                results.append(None)
        return results

    def get_chat_completions(self, prompts: list[str]) -> list[Optional[str]]:
        """Get chat completions for a list of prompts (synchronously)."""
        results = []
        for prompt in prompts:
            try:
                result = self.get_chat_completion(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Chat completion failed for prompt: {prompt}. Error: {e}")
                results.append(None)
        return results


class HEALpacaAsyncClient:
    def __init__(
            self,
            chat_model=CHAT_MODEL,
            embedding_model=EMBEDDING_MODEL,
            api_url=LLM_API_URL,
            embedding_url=EMBEDDING_URL,
            chat_temperature=TEMPERATURE,
            max_concurrent_requests: int = None,
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.api_url = api_url
        self.embedding_url = embedding_url
        self.chat_temperature = chat_temperature
        self.headers = headers
        if max_concurrent_requests is None:
            max_concurrent_requests = 2 if USE_LOCAL else 5
        self.semaphore = Semaphore(max_concurrent_requests)

    async def _post(self, url: str, model: str, prompt: str, max_retries: int = 3) -> list[float] | str | None:
        """
            Send POST request with progressive timeout and retry logic
        """

        for attempt in range(max_retries):
            try:
                base_timeout = 600.0
                current_timeout = base_timeout * (1 + attempt * 0.5)

                timeout = httpx.Timeout(
                    connect=30.0,
                    read=current_timeout,
                    write=30.0,
                    pool=30.0
                )

                async with httpx.AsyncClient(
                        timeout=timeout,
                        limits=httpx.Limits(max_connections=3, max_keepalive_connections=1)
                ) as client:

                    response = await client.post(
                        url,
                        json=make_payload(model, prompt, self.chat_temperature),
                        headers=self.headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    result = data.get("embedding") or data.get("response")

                    if result:
                        return result
                    else:
                        logger.warning(f"Empty response on attempt {attempt + 1}")

            except httpx.ReadTimeout as e:
                logger.warning(f"Read timeout on attempt {attempt + 1}/{max_retries} (timeout: {current_timeout:.1f}s)")

            except httpx.ConnectTimeout as e:
                logger.warning(f"Connect timeout on attempt {attempt + 1}/{max_retries}")

            except httpx.RemoteProtocolError as e:
                logger.warning(f"Server disconnected on attempt {attempt + 1}/{max_retries}: {e}")

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code >= 500:
                    logger.warning(f"Server error {status_code} on attempt {attempt + 1}/{max_retries}")
                else:
                    logger.error(f"Client error {status_code}, not retrying")
                    return None

            except httpx.RequestError as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{max_retries}: {str(e)}")

            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {e}")

            if attempt < max_retries - 1:
                base_delay = 2 ** attempt
                jitter = random.uniform(0.8, 1.2)
                delay = min(base_delay * jitter, 30)
                logger.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

        logger.error(f"All {max_retries} attempts failed for {model} request")
        return None

    async def get_embedding(self, text: str) -> list[float] | None:
        return await self._post(self.embedding_url, self.embedding_model, text)

    async def get_chat_completion(self, prompt: str) -> str | None:
        return await self._post(self.api_url, self.chat_model, prompt)

    async def throttled_chat_completion(self, prompt: str) -> str | None:
        async with self.semaphore:
            return await self.get_chat_completion(prompt)

    async def get_async_embeddings(self, texts: list[str]) -> tuple[Optional[list[float]], ...]:
        return await asyncio.gather(*(self.get_embedding(text) for text in texts), return_exceptions=True)

    async def get_async_chat_completions(self, prompts: list[str]) -> list[Optional[str]]:
        tasks = [self.throttled_chat_completion(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)


