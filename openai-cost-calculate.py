
"""
Configuration management for UGC Machine.
Uses pydantic-settings for environment-based configuration.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""
    kie_api_key: str = ""

    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000

    # Output Directory
    output_dir: str = "output_5"

    # LLM Provider for prompt generation: "gemini" or "openai"
    prompt_generator_provider: Literal["gemini", "openai"] = "gemini"

    # Model Configuration
    gemini_model: str = "gemini-2.5-flash"  # For video analysis
    prompt_generator_model: str = "gemini-2.5-flash"  # For prompt generation (when using Gemini)
    openai_model: str = "gpt-4o-mini"  # For prompt generation (when using OpenAI)

    # Database Configuration
    postgres_user: str = "postgres"
    postgres_password: str = "xyz123"
    postgres_db: str = "video_clone"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Gemini Cost Tracking Rates (per 1M tokens)
    gemini_cost_input_text_image_video: float = 0.30
    gemini_cost_input_audio: float = 1.00
    gemini_cost_output: float = 2.50

    # KIE Flat-Rate Pricing (per video, by model × resolution × duration)
    kie_cost_sora2_720p_10s: float = 0.15
    kie_cost_sora2_720p_15s: float = 0.175
    kie_cost_sora2_pro_720p_10s: float = 0.75
    kie_cost_sora2_pro_720p_15s: float = 1.35
    kie_cost_sora2_pro_1080p_10s: float = 1.65
    kie_cost_sora2_pro_1080p_15s: float = 3.15

    # OpenAI Pricing (per 1K tokens)
    openai_cost_gpt4o_prompt: float = 0.00250
    openai_cost_gpt4o_completion: float = 0.01000
    openai_cost_gpt41_mini_prompt: float = 0.000400
    openai_cost_gpt41_mini_completion: float = 0.001600
    openai_cost_gpt4o_mini_prompt: float = 0.000150
    openai_cost_gpt4o_mini_completion: float = 0.000600
    openai_cost_gpt5_mini_prompt: float = 0.00025
    openai_cost_gpt5_mini_completion: float = 0.00200
    openai_cost_sora2_per_second: float = 0.10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def output_path(self) -> Path:
        """Get the output directory as a Path object."""
        return Path(self.output_dir)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

# OpenAI model → (settings prompt attr, settings completion attr)
_OPENAI_MODEL_PRICING_MAP = {
    "gpt-4o": ("openai_cost_gpt4o_prompt", "openai_cost_gpt4o_completion"),
    "gpt-4.1-mini": ("openai_cost_gpt41_mini_prompt", "openai_cost_gpt41_mini_completion"),
    "gpt-4o-mini": ("openai_cost_gpt4o_mini_prompt", "openai_cost_gpt4o_mini_completion"),
    "gpt-5-mini": ("openai_cost_gpt5_mini_prompt", "openai_cost_gpt5_mini_completion"),
}


def calculate_openai_cost(completion, model: str) -> float:
    """
    Calculate the cost of an OpenAI API call.

    Args:
        completion: The OpenAI response object (must have .usage attribute).
        model: The OpenAI model name (e.g. 'gpt-4o', 'gpt-4.1-mini-2025-04-14').

    Returns:
        float: The calculated cost in USD.
    """
    try:
        if not model:
            logger.warning(f"[COST] No OpenAI model specified. Cost will be 0.")
            return 0.0

        # Direct model match
        if model not in _OPENAI_MODEL_PRICING_MAP:
            logger.warning(f"[COST] Unknown OpenAI model: {model}. Cost will be 0.")
            return 0.0

        prompt_attr, completion_attr = _OPENAI_MODEL_PRICING_MAP[model]
        prompt_rate = getattr(settings, prompt_attr, 0.0)
        completion_rate = getattr(settings, completion_attr, 0.0)

        usage = completion.usage
        if not usage:
            logger.warning("[COST] No usage data available from OpenAI response.")
            return 0.0

        # Support for both new (input/output) and old (prompt/completion) attribute names
        prompt_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
        completion_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))

        prompt_cost = (prompt_tokens / 1000) * prompt_rate
        completion_cost = (completion_tokens / 1000) * completion_rate
        total_cost = prompt_cost + completion_cost

        logger.info(
            f"Calculated OpenAI Cost: ${total_cost:.6f} "
            f"(model={model}, prompt={prompt_tokens}tok/${prompt_cost:.6f}, "
            f"completion={completion_tokens}tok/${completion_cost:.6f})"
        )
        return total_cost

    except Exception as e:
        logger.error(f"[COST] Error calculating OpenAI cost: {e}")
        return 0.0


#--------------------------------------------------------------------------------------------

def calculate_openai_video_cost(duration_seconds: int) -> float:
    """
    Calculates cost for OpenAI's Sora-2 model video generation.
    It costs $0.10 per second based on the duration of the video.

    Args:
        duration_seconds: Video duration in seconds.

    Returns:
        float: Calculated total cost.
    """
    try:
        if not duration_seconds or int(duration_seconds) <= 0:
            logger.warning(f"[COST] Invalid duration {duration_seconds} for Sora generation cost calculation.")
            return 0.0

        rate = getattr(settings, 'openai_cost_sora2_per_second', 0.10)
        cost = float(duration_seconds) * rate

        logger.info(f"Calculated OpenAI Sora Video Cost: ${cost:.6f} for {duration_seconds} seconds")
        return cost

    except Exception as e:
        logger.error(f"[COST] Error calculating OpenAI Sora video cost: {e}")
        return 0.0
 