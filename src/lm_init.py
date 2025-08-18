import os
import dspy
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
import litellm
from .logging_setup import logger
from .config import config

load_dotenv()

@retry(wait=wait_fixed(4), stop=stop_after_attempt(10))
def _retry_call(func, *args, **kwargs):
    return func(*args, **kwargs)

class RateLimitedLM(dspy.LM):
    def __init__(self, model, api_key, **kwargs):
        if not api_key:
            raise ValueError(f"API key is not provided for model: {model}")
        super().__init__(model=model, api_key=api_key, **kwargs)

    def __call__(self, prompt=None, messages=None, **kwargs):
        if prompt is not None:
            return _retry_call(super().__call__, prompt=prompt, **kwargs)
        elif messages is not None:
            return _retry_call(super().__call__, messages=messages, **kwargs)
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    def litellm_completion(self, request, num_retries, cache):
        return litellm.completion(request=request, num_retries=num_retries)

def init_lms():
    try:
        main_api_key = os.getenv("GEMINI_API_KEY")
        judge_api_key = os.getenv("GEMINI_API_KEY")
        lm = RateLimitedLM(config.main_model, temperature=config.main_temperature, api_key=main_api_key)
        judge_lm = RateLimitedLM(config.judge_model, temperature=config.judge_temperature, api_key=judge_api_key)
        dspy.settings.configure(lm=lm)
        return lm, judge_lm
    except ValueError as e:
        logger.error(f"API Key Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing Language Models: {e}")
        raise
