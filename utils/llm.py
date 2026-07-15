# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Dict, List, Union

from openai import OpenAI


class LLMClient:
    """LLM client using OpenAI API for GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o",
        human_model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: float = None,
        max_retries: int = None,
    ):
        """Initialize the LLM client.

        Args:
            model: Model name for agent (e.g., "gpt-4o", "gpt-4o-mini", "gpt-4-turbo")
            human_model: Model name for human simulation (if None, uses same as model)
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY environment variable)
            base_url: OpenAI-compatible base URL
            timeout: Request timeout in seconds
            max_retries: Maximum OpenAI client retries
        """
        env_model = os.getenv("PAHF_AGENT_MODEL") or os.getenv("OPENAI_MODEL")
        env_human_model = os.getenv("PAHF_HUMAN_MODEL")
        self.model = env_model or model
        self.human_model = human_model or env_human_model or self.model

        # Initialize OpenAI client
        api_key = (
            api_key
            or os.getenv("PAHF_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it via api_key parameter "
                "or set PAHF_OPENAI_API_KEY / OPENAI_API_KEY environment variable."
            )

        base_url = (
            base_url
            or os.getenv("PAHF_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        timeout = timeout or float(os.getenv("PAHF_OPENAI_TIMEOUT", "180"))
        max_retries = max_retries or int(os.getenv("PAHF_OPENAI_MAX_RETRIES", "5"))

        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def build_msgs(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Build messages in OpenAI format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            List of message dicts in OpenAI format
        """
        formatted_messages = []
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )
        return formatted_messages

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        use_human_model: bool = False,
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> str:
        """Generate text using OpenAI API.

        Args:
            prompt: Either a string (user message) or list of message dicts
            use_human_model: Whether to use human model instead of agent model
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Generated text string
        """
        # Build messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = self.build_msgs(prompt)

        # Choose model
        model = self.human_model if use_human_model else self.model

        MAX_ATTEMPTS = 5

        for i in range(MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

                # Return the generated text
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e).lower()

                # Check if this is a rate limit error
                if "rate" in error_str or "limit" in error_str:
                    if i < MAX_ATTEMPTS - 1:
                        wait_time = 5 * (i + 1)  # 5, 10, 15, 20 seconds
                        print(
                            f"Rate limited, waiting {wait_time}s (attempt {i+1}/{MAX_ATTEMPTS})"
                        )
                        time.sleep(wait_time)
                    else:
                        print("All retries failed due to rate limiting")
                        raise
                else:
                    # Other errors - retry with exponential backoff
                    if i < MAX_ATTEMPTS - 1:
                        wait_time = 2 * (i + 1)  # 2, 4, 6, 8 seconds
                        print(
                            f"Error: {e}, retrying in {wait_time}s... (attempt {i+1}/{MAX_ATTEMPTS})"
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"All retries failed: {e}")
                        raise

        # This should never be reached, but just in case
        raise Exception(
            "Unexpected error: all attempts completed without success or exception"
        )
