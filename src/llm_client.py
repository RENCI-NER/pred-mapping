import requests
import aiohttp
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
            chat_model: str = "llama3.1:latest",  #"HEALpaca-2.0",
            embedding_model: str = "nomic-embed-text",
            api_url: str = "https://healpaca.apps.renci.org/api/generate",
            #"https://healpaca.apps.renci.org/api/generate",
            embedding_url: str = "https://healpaca.apps.renci.org/api/embeddings",
            chat_temperature: float = 0.5,
    ):
        super().__init__(chat_model=chat_model, embedding_model=embedding_model, chat_temperature=chat_temperature)
        self.api_url = api_url
        self.embedding_url = embedding_url

    def get_embedding(self, text: str) -> list:
        return _cached_embedding_request(text)

    def get_chat_completion(self, prompt: str):
        """ Get single chat response """
        request = self.chat_request(prompt)
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, json=request, headers=headers)
        print(response)
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


class AsyncHEALpacaClient:
    def __init__( self, chat_model: str = "llama3.1:latest",
                  api_url: str = "https://ollama.apps.renci.org/api/generate" ):
        self.chat_model = chat_model
        self.api_url = api_url

    async def get_chat_completion( self, prompt: str ) -> str:
        """ Get a chat completion using the aiohttp session """
        async with aiohttp.ClientSession() as session:  # Creating a new session for each request
            request_data = {
                "model": self.chat_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,
            }
            async with session.post(self.api_url, json=request_data) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        return data.get("response", "")
                    except Exception as e:
                        print(f"Error while parsing response: {e}")
                        return "Error parsing response"
                else:
                    print(f"Request failed with status: {response.status}-{Exception}")
                    return "Request failed"
