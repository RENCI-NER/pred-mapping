import os
import requests
import asyncio
import httpx
from dotenv import load_dotenv
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "https://healpaca.apps.renci.org/api/generate")
CHAT_MODEL = os.getenv("CHAT_MODEL", "HEALpaca-2.0")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.5))
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "https://healpaca.apps.renci.org/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
headers = {"Content-Type": "application/json"}


def make_payload(model: str, prompt: str, temperature: float) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }


@lru_cache(maxsize=2048)
def _cached_embedding_request( text: str ) -> list[float] | None:
    request = make_payload(EMBEDDING_MODEL, text, TEMPERATURE)
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(EMBEDDING_URL, json=request, headers=headers)
            response.raise_for_status()
            return response.json().get("embedding")
    except httpx.HTTPError as e:
        logger.error(f"Cached embedding request failed: {e}")
        return None


class LLMClient:
    def __init__(
            self,
            embedding_model: str = None,
            chat_model: str = None,
            chat_temperature: float = TEMPERATURE
    ):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.chat_temperature = chat_temperature

    def embedding_request( self, text: str ):
        """ Create an API embedding request for input text. """
        request_content = make_payload(self.embedding_model, text, self.chat_temperature)
        return request_content

    def chat_request( self, prompt: str ):
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

    def get_chat_completion(self, prompt: str):
        """ Get single chat response """
        request = self.chat_request(prompt)
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, json=request, headers=headers)
                response.raise_for_status()
                return response.json().get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Chat Completion request failed: {e}")
            return None

    # async def get_async_embedding(self, texts: list[str]) -> list[list[float] | None]:
    #     tasks = [asyncio.create_task(self.get_embedding(text)) for text in texts]
    #     return await asyncio.gather(*tasks)
    #
    # async def get_async_chat_completion(self, prompts: list[str]) -> list[str | None]:
    #     tasks = [asyncio.create_task(self.get_chat_completion(prompt)) for prompt in prompts]
    #     return await asyncio.gather(*tasks)


class HEALpacaAsyncClient:
    def __init__(
        self,
        chat_model=CHAT_MODEL,
        embedding_model=EMBEDDING_MODEL,
        api_url=LLM_API_URL,
        embedding_url=EMBEDDING_URL,
        chat_temperature=TEMPERATURE,
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.api_url = api_url
        self.embedding_url = embedding_url
        self.chat_temperature = chat_temperature
        self.headers = headers

    async def _post(self, url: str, model: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    url,
                    json=make_payload(model, prompt, self.chat_temperature),
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("embedding") or data.get("response")
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"[HTTP ERROR] {url} returned status {status_code}")
                raise RuntimeError(f"HTTP {status_code} error calling {url}")
                return None

            except httpx.RequestError as e:
                logger.error(f"[REQUEST ERROR] Could not reach {url} for model '{model}': {str(e)}")
                return None

            except Exception as e:
                logger.exception(f"[UNEXPECTED ERROR] calling {url} with model '{model}': {str(e)}")
                return None

    async def get_embedding(self, text: str) -> list[float] | None:
        return await self._post(self.embedding_url, self.embedding_model, text)

    async def get_chat_completion(self, prompt: str) :
        return await self._post(self.api_url, self.chat_model, prompt)

    async def get_async_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        return await asyncio.gather(*(self.get_embedding(text) for text in texts))

    async def get_async_chat_completions(self, prompts: list[str]) -> list[str | None]:
        return await asyncio.gather(*(self.get_chat_completion(prompt) for prompt in prompts))
