import json
import asyncio
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


def load_from_json(json_file_path):
    """
    Safely load a json file from disk or return error

    Args:
        json_file_path:
    """

    try:
        with open(json_file_path, "r") as f:
            json_file = json.load(f)
        # logger.info(f"Successfully loaded JSON with {len(json_file)} top-level entries")
        return json_file
    except FileNotFoundError:
        logger.error(f"File not found: {json_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        raise


def chunked(iterable, n: int = 25):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


async def safe_limited_embedding(async_client, prompts: list[str], retries: int = 2, backoff: float = 3.0):
    """
    Safely get embeddings with retry logic and exponential backoff.
    IMPROVED: Reduced retries and faster backoff for unstable server

    Args:
        async_client: The async client for embeddings
        prompts: List of text strings to embed (typically 25 or fewer)
        retries: Number of retry attempts (reduced from 3 to 2)
        backoff: Base delay for exponential backoff (reduced from 5.0 to 3.0)

    Returns:
        List of embeddings or list of None values if all retries fail
    """
    for attempt in range(1, retries + 1):
        try:
            return await async_client.get_async_embeddings(prompts)
        except Exception as e:
            logger.warning(f"[Retry {attempt}/{retries}] Failed embedding batch: {e}")
            if attempt < retries:
                await asyncio.sleep(backoff * attempt)  # exponential backoff
            else:
                # Final fallback: return list of None placeholders
                logger.error(f"All retries exhausted for batch of {len(prompts)} texts")
                return [None] * len(prompts)


async def safe_limited_chat_completion(async_client, prompts: list[str], retries: int = 2, backoff: float = 2.0):
    """
    IMPROVED: Better handling for unstable server with individual prompt retry

    Args:
        async_client: The async client for chat completions
        prompts: List of prompts for chat completion
        retries: Number of retry attempts per prompt (reduced from 3 to 2)
        backoff: Base delay for exponential backoff (reduced from 5.0 to 2.0)

    Returns:
        List of chat completions or None values for failed prompts
    """
    results = [None] * len(prompts)

    # Track which prompts still need processing
    pending_indices = list(range(len(prompts)))

    for attempt in range(1, retries + 1):
        if not pending_indices:
            break

        logger.info(f"[Attempt {attempt}/{retries}] Processing {len(pending_indices)} prompts")

        # Create tasks only for pending prompts
        tasks = []
        current_indices = []

        for idx in pending_indices:
            tasks.append(async_client.get_chat_completion(prompts[idx]))
            current_indices.append(idx)

        try:
            # Process with timeout to avoid hanging
            batch_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=900.0  # 15 minutes total timeout per batch
            )

            # Update results and track what still needs processing
            new_pending = []
            for idx, result in zip(current_indices, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Prompt {idx} failed attempt {attempt}: {result}")
                    new_pending.append(idx)
                elif result is not None:
                    results[idx] = result
                    logger.debug(f"✓ Prompt {idx} completed")
                else:
                    logger.warning(f"Prompt {idx} returned None on attempt {attempt}")
                    new_pending.append(idx)

            pending_indices = new_pending

        except asyncio.TimeoutError:
            logger.error(f"Batch timeout on attempt {attempt} - {len(pending_indices)} prompts timed out")

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}: {e}")

        # Backoff between retry attempts
        if attempt < retries and pending_indices:
            delay = backoff * attempt
            logger.info(f"Waiting {delay}s before retry...")
            await asyncio.sleep(delay)

    # Log final results
    successful = sum(1 for r in results if r is not None)
    failed = len(results) - successful
    logger.info(f"Completed: {successful}/{len(results)} prompts ({failed} failed)")

    return results


# ALTERNATIVE: Even more conservative version for very unstable servers
async def ultra_safe_chat_completion(async_client, prompts: list[str], delay_between_requests: float = 1.0):
    """
    Ultra-conservative approach: process one request at a time with delays
    Use this if the batch approach still causes server issues
    """
    results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i + 1}/{len(prompts)}")

        # Try each prompt with built-in retries from the client
        try:
            result = await async_client.get_chat_completion(prompt)
            results.append(result)

            if result is None:
                logger.warning(f"Prompt {i + 1} returned None")
            else:
                logger.debug(f"Prompt {i + 1} completed successfully")

        except Exception as e:
            logger.error(f"Prompt {i + 1} failed with exception: {e}")
            results.append(None)

        # Small delay between requests to be nice to server
        if i < len(prompts) - 1:  # Don't delay after the last request
            await asyncio.sleep(delay_between_requests)

    return results
