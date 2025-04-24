import requests
import asyncio
import httpx
from functools import lru_cache


@lru_cache(maxsize=2048)
def _cached_embedding_request( text: str ) -> list:
    import requests
    request = {
        "model": "nomic-embed-text",
        "prompt": text,
        "stream": False,
        "temperature": 0.5
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post("https://healpaca.apps.renci.org/api/embeddings", json=request, headers=headers)

    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        print(Exception(f"Error Code: {response.status_code}"))
        return None


class LLMClient:
    def __init__(
            self,
            embedding_model: str = None,
            chat_model: str = None,
            chat_temperature: float = 0.5
    ):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.chat_temperature = chat_temperature

    def embedding_request( self, text: str ):
        """ Create an API embedding request for input text. """
        request_content = {
            "model": self.embedding_model,
            "prompt": text,
            "stream": False,
            "temperature": self.chat_temperature,
        }
        return request_content

    def chat_request( self, prompt: str ):
        """ Create an API chat request from system and user prompts. """
        request_content = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.chat_temperature
        }
        return request_content


class HEALpacaClient(LLMClient):
    def __init__(
            self,
            chat_model: str = "HEALpaca-2.0", #"llama3.1:latest"
            embedding_model: str = "nomic-embed-text",
            api_url: str = "https://healpaca.apps.renci.org/api/generate",
            #"https://ollama.apps.renci.org/api/generate",
            embedding_url: str = "https://healpaca.apps.renci.org/api/embeddings",
            chat_temperature: float = 0.5,
    ):
        super().__init__(chat_model=chat_model, embedding_model=embedding_model, chat_temperature=chat_temperature)
        self.api_url = api_url
        self.embedding_url = embedding_url


    def get_embedding(self, text: str) -> list:
        return _cached_embedding_request(text)

    async def get_async_embedding(self, texts: list) -> list:
        tasks = [asyncio.create_task(self.get_embedding(text)) for text in texts]
        return await asyncio.gather(*tasks)

    def get_chat_completion(self, prompt: str):
        """ Get single chat response """
        request = self.chat_request(prompt)
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, json=request, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json().get("response", "")
                return data
            except Exception as e:
                print(f"Error Code: {response.status_code}")
                return response
        else:
            print(f"Exception Error {response.status_code}")
            return None

    async def get_async_chat_completion(self, prompts: list):
        task = [asyncio.create_task(self.get_chat_completion(prompt)) for prompt in prompts]
        return await asyncio.gather(*task)


class HEALpacaAsyncClient:
    def __init__(
        self,
        chat_model="HEALpaca-2.0",
        embedding_model="nomic-embed-text",
        api_url="https://healpaca.apps.renci.org/api/generate",
        embedding_url="https://healpaca.apps.renci.org/api/embeddings",
        chat_temperature=0.5,
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.api_url = api_url
        self.embedding_url = embedding_url
        self.chat_temperature = chat_temperature
        self.headers = {"Content-Type": "application/json"}

    async def _post(self, url: str, model: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    url,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.chat_temperature,
                    },
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("embedding") or data.get("response")
            except Exception as e:
                print(f"Request failed to {url}: {e}")
                return None

    async def get_embedding(self, text: str):
        return await self._post(self.embedding_url, self.embedding_model, text)

    async def get_chat_completion(self, prompt: str):
        return await self._post(self.api_url, self.chat_model, prompt)

    async def get_async_embeddings(self, texts: list[str]):
        return await asyncio.gather(*(self.get_embedding(text) for text in texts))

    async def get_async_chat_completions(self, prompts: list[str]):
        return await asyncio.gather(*(self.get_chat_completion(prompt) for prompt in prompts))
