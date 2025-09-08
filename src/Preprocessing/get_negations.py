import re
import json
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
import asyncio
from src.llm_client import HEALpacaClient, HEALpacaAsyncClient
from src.utils import safe_limited_chat_completion

DEFAULT = "NOT ENOUGH INFORMATION"
BAD = ["negation_of_the_descriptor_text", "", "none", "None"]


def get_prompt(descriptor_text):
    negation_system_prompt = f"""
    You are a biomedical researcher extracting negations of ontological predicates.

    Your Task:
        Given a description, return its natural negation.

    Rules:
        1. Preserve the meaning but negate the entire description.
        3. Do not summarize or change the structure of the descriptor text.
        4. If there is not enough information to create a negation, your response should be "NOT ENOUGH INFORMATION"
        4. Only return the negation—no explanations or extra text.

    Examples:
        - "has effect" → "does not have effect"
        - "during which ends" → "during which does not ends"
        - "happens during" → "does not happen during"
        - "has boundary" → "does not have a boundary"
        - "characteristic of" → "is not characteristic of"
        - "X happens_during Y iff..." → "X does not happen_during Y iff..."
        - "m has_muscle_origin s iff m is attached_to s, and it is the case that when m contracts, s does not move. The site of the origin tends to be more proximal and have greater mass than what the other end attaches to." → "m does not have_muscle_origin s iff m is not attached_to s, and it is not the case that when m contracts, s does not move. The site of the origin does not tend to be more proximal and have greater mass than what the other end attaches to."
        - "A relationship between a disease and an anatomical entity where the disease has one or more features that are located in that entity."  → "A relationship between a disease and an anatomical entity where the disease does not have one or more features that are located in that entity."

    Input:
        "{descriptor_text}"

   
    Output:
        A JSON object with these exact keys and format:
        {{"negation_of_the_descriptor_text": "negated version of the input descriptor_text" if a good match exists, otherwise "NOT ENOUGH INFORMATION"}}

    Do not include any other output or explanation. Only output the JSON object.

    """
    return negation_system_prompt


class NegationClient(HEALpacaClient):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def get_negation(self, descriptor: str):
        """ Send options for a single relationship to OpenAI LLM """
        prompt = get_prompt(descriptor)
        return split_negated_descriptor_response(self.get_chat_completion(prompt))

    async def get_async_negations(self, descriptors: list[str], batch_size: int = 10) -> list[str]:
        """
            Processes descriptor negations
        """
        async_client = HEALpacaAsyncClient(max_concurrent_requests=3)
        all_results = []

        total_batches = (len(descriptors) + batch_size - 1)

        for i in range(0, len(descriptors), batch_size):
            batch_num = (i // batch_size) + 1
            print(
                f"Processing batch {batch_num}/{total_batches} (items {i + 1}–{min(i + batch_size, len(descriptors))} of {len(descriptors)})")

            batch = descriptors[i:i + batch_size]
            prompts = [get_prompt(d) for d in batch]

            try:
                raw_responses = await safe_limited_chat_completion(async_client, prompts, retries=2)

                results = []
                for j, resp in enumerate(raw_responses):
                    if resp is None:
                        print(f"  Warning: Response {j + 1} in batch {batch_num} was None")
                        results.append(DEFAULT)
                    elif isinstance(resp, Exception):
                        print(f"  Error: Response {j + 1} in batch {batch_num} had exception: {resp}")
                        results.append(DEFAULT)
                    elif isinstance(resp, str):
                        processed = split_negated_descriptor_response(resp.strip())
                        results.append(processed)
                    else:
                        print(f"  Warning: Response {j + 1} in batch {batch_num} had unexpected type: {type(resp)}")
                        results.append(DEFAULT)

                all_results.extend(results)

                successful = sum(1 for r in results if r != DEFAULT)
                print(f"  Batch {batch_num} completed: {successful}/{len(results)} successful")

            except Exception as e:
                print(f"  Error: Batch {batch_num} failed completely: {e}")
                all_results.extend([DEFAULT] * len(batch))

            if batch_num < total_batches:
                print(f"  Pausing 2s before next batch...")
                await asyncio.sleep(2.0)

        return all_results


def split_negated_descriptor_response(response: str) -> str:
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and parsed:
            value = next(iter(parsed.values()))
            if not isinstance(value, str):
                return DEFAULT
            value = value.strip()
            if isinstance(value, str):
                value = value.strip()
                if value.lower() in BAD or len(value.split()) < 3:
                    return DEFAULT
                return value
        elif isinstance(parsed, str):
            value = parsed.strip()
            if len(value.split()) < 3:
                return DEFAULT
            return value
    except RuntimeError:
        pass
    except json.JSONDecodeError:
        pass

    if re.search(r'NOT ENOUGH INFORMATION', response, re.IGNORECASE):
        return DEFAULT

    match = re.search(r'\{\s*"[^"]*"\s*:\s*"([^"]+)"\s*\}', response)
    if match:
        value = match.group(1).strip()
        if value.lower() in BAD:
            return DEFAULT
        return value

    match = re.search(r'"([^"]+)"', response)
    if match:
        value = match.group(1).strip()
        if len(value.split()) < 3:
            return DEFAULT
        return value

    for line in response.splitlines():
        line = line.strip()
        if len(line.split()) >= 3:
            return line

    return DEFAULT


async def process_descriptors(input_json_path, output_json_path):
    start_time = time.time()
    client = NegationClient()

    with open(input_json_path, 'r') as infile:
        data = json.load(infile)

    # Process descriptors
    negated_data = defaultdict(list)
    total_predicates = len(data)
    processed_predicates = 0

    for predicate, descriptors in tqdm(data.items(), desc="Processing predicates"):
        print(f"\nProcessing predicate: {predicate} ({len(descriptors)} descriptors)")

        try:
            # Use smaller batch size for more stable processing
            negated_descriptors = await client.get_async_negations(descriptors, batch_size=3)
            negated_data[f"{predicate} NEG"] = negated_descriptors

            # Log success rate for this predicate
            successful = sum(1 for nd in negated_descriptors if nd != DEFAULT)
            print(f"Predicate {predicate} completed: {successful}/{len(descriptors)} successful negations")

        except Exception as e:
            print(f"Error processing predicate {predicate}: {e}")
            # Add default responses for failed predicate
            negated_data[f"{predicate} NEG"] = [DEFAULT] * len(descriptors)

        processed_predicates += 1
        print(f"Overall progress: {processed_predicates}/{total_predicates} predicates completed")

        # # Save intermediate results every 5 predicates
        # if processed_predicates % 5 == 0:
        #     temp_output = output_json_path.replace('.json', f'_temp_{processed_predicates}.json')
        #     with open(temp_output, 'w') as temp_file:
        #         json.dump(negated_data, temp_file, indent=2)
        #     print(f"Saved intermediate results to {temp_output}")

    with open(output_json_path, 'w') as outfile:
        json.dump(negated_data, outfile, indent=2)

    elapsed_time = time.time() - start_time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.1f} minutes)")

    total_descriptors = sum(len(descriptors) for descriptors in negated_data.values())
    successful_negations = sum(1 for descriptors in negated_data.values()
                               for desc in descriptors if desc != DEFAULT)
    print(
        f"Final Results: {successful_negations}/{total_descriptors} successful negations ({100 * successful_negations / total_descriptors:.1f}%)")


if __name__ == "__main__":
    # Paths to the input and output JSON files
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="chemprot_mappings.json", help="Mappings file")
    parser.add_argument("-n", "--negations", default="negated_chemprot_mappings.json",
                        help="Negation mappings file")
    args = parser.parse_args()

    # Process the descriptors to generate negated versions
    asyncio.run(process_descriptors(args.mappings, args.negations))

    print(f"Negated descriptors have been saved to {args.negations}")
