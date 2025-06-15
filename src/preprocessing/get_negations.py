import os
import re
import json
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
import httpx, asyncio
from src.llm_client import HEALpacaClient


def get_prompt( descriptor_text ):
    negation_system_prompt = f"""
    You are a biomedical researcher extracting negations of ontological predicates.

    ## Task:
    Given a description, return its natural negation. If there is not enough information to create a negation, your response should be {"NOT ENOUGH INFORMATION"}

    ## Rules:
    1. Preserve the meaning but negate the entire description.
    2. Do **not** use synonyms or antonyms.
    3. Do **not** summarize or change the structure.
    4. Only return the negation—no explanations or extra text.

    ## Examples:
    - "has effect" → "does not have effect"
    - "during which ends" → "during which does not ends"
    - "happens during" → "does not happen during"
    - "has boundary" → "does not have a boundary"
    - "characteristic of" → "is not characteristic of"
    - "X happens_during Y iff..." → "X does not happen_during Y iff..."
    - "m has_muscle_origin s iff m is attached_to s, and it is the case that when m contracts, s does not move. The site of the origin tends to be more proximal and have greater mass than what the other end attaches to." → "m does not have_muscle_origin s iff m is not attached_to s, and it is not the case that when m contracts, s does not move. The site of the origin does not tend to be more proximal and have greater mass than what the other end attaches to."
    - "A relationship between a disease and an anatomical entity where the disease has one or more features that are located in that entity."  → "A relationship between a disease and an anatomical entity where the disease does not have one or more features that are located in that entity."

    ## Input:
    "{descriptor_text}"

    ## Make sure your response strictly ends with a json object:
    
    {{"negation of the descriptor_text or NOT ENOUGH INFORMATION in capital letters If there is not enough information to create a negation"}}
    """
    return negation_system_prompt


class NegationClient(HEALpacaClient):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def get_negation( self, descriptor: str ):
        """ Send options for a single relationship to OpenAI LLM """
        prompt = get_prompt(descriptor)
        return split_negated_descriptor_response(self.get_chat_completion(prompt))

    async def get_async_negations( self, descriptors: list[str] ):
        """ Get negations asynchronously """

        max_conns = 10
        limits = httpx.Limits(max_connections=max_conns, max_keepalive_connections=5)
        async with httpx.AsyncClient(timeout=httpx.Timeout(max_conns * 60), limits=limits) as asynclient:
            tasks = [asyncio.create_task(self.get_async_chat_completion(get_prompt(descriptor), asynclient=asynclient, max_conns=max_conns)) for descriptor in descriptors]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        negations = []
        for response in responses:
            if isinstance(response, Exception):
                print(f"Error in request: {response}")
                negations.append("ERROR")
            else:
                negations.append(split_negated_descriptor_response(response))

        return negations

    # async def get_async_negations( self, descriptors: list[str]):
    #     tasks = []
    #     max_conns = 10
    #     connector = aiohttp.TCPConnector(limit=0)  # need unlimited connections
    #     async with aiohttp.ClientSession(connector=connector) as asynclient:
    #     # limits = httpx.Limits(max_connections=max_conns, max_keepalive_connections=5)
    #     # async with httpx.AsyncClient(timeout=httpx.Timeout(max_conns * 60), limits=limits) as asynclient:
    #         for descriptor in descriptors:
    #             prompt = get_prompt(descriptor)
    #             tasks.append(asyncio.create_task(self.get_async_chat_completion(prompt, asynclient=asynclient, max_conns=max_conns)))
    #         responses = await asyncio.gather(*tasks)
    #     negations = [split_negated_descriptor_response(response.json()["response"]) for response in responses]
    #     return negations
    # TODO: Process batch results from job ID


def split_negated_descriptor_response( negated_descriptor: str ) -> str:
    """Extracts the occurrence of text inside curly braces and returns it as a string."""
    match = re.search(r'\{(.*?)\}', negated_descriptor)
    return match.group(1) if match else ""


async def process_descriptors(input_json_path, output_json_path):
    start_time = time.time()

    client = NegationClient()

    # Load JSON file
    with open(input_json_path, 'r') as infile:
        data = json.load(infile)

    # Process descriptors
    negated_data = defaultdict(list)
    for predicate, descriptors in tqdm(data.items(), desc="Processing predicates"):
        negated_descriptors = await client.get_async_negations(descriptors)
        negated_data[f"{predicate} NEG"] = negated_descriptors

    # Save results
    with open(output_json_path, 'w') as outfile:
        json.dump(negated_data, outfile, indent=2)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")  # Print elapsed time


# async def process_descriptors( input_json_path, output_json_path ):
#     client = NegationClient()
#     print('client', client)
#     # Load the JSON file
#     with open(input_json_path, 'r') as infile:
#         data = json.load(infile)
#
#     # Process each descriptor to create negated versions
#     negated_data = defaultdict(list)
#     for predicate, descriptors in tqdm(data.items(), desc="Predicates", position=0):
#         negated_descriptors = await client.get_async_negations(descriptors)
#         # for negated_descriptor in tqdm(negated_descriptors, desc="Descriptors", position=1):
#         #     if "}" in negated_descriptor:
#         #         negated_descriptor = split_negated_descriptor_response(negated_descriptor)
#         negated_data[f"{predicate} NEG"] = negated_descriptors
#
#     # Save the negated descriptors to a new JSON file
#     with open(output_json_path, 'w') as outfile:
#         json.dump(negated_data, outfile, indent=2)


# TODO: Parse batch results


if __name__ == "__main__":
    # Paths to the input and output JSON files
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="biolink_mappings.json", help="Mappings file")
    parser.add_argument("-n", "--negations", default="negated_biolink_mappings_llama.json",
                        help="Negation mappings file")
    args = parser.parse_args()
    input_json_path = args.mappings
    output_json_path = args.negations

    # Process the descriptors to generate negated versions
    asyncio.run(process_descriptors(input_json_path, output_json_path))

    print(f"Negated descriptors have been saved to {output_json_path}")
