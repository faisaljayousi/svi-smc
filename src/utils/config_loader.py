from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class SVIParams(BaseModel):
    """Schema for SVI bounds/initial guesses."""

    lower: List[float]
    upper: List[float]

    @field_validator("lower", "upper")
    @classmethod
    def check_length(cls, v):
        if len(v) != 5:
            raise ValueError(
                "SVI bounds must have exactly 5 elements: [a, b, rho, m, sigma]"
            )
        return v


class ModelConfig(BaseModel):
    name: str
    initial_guess: List[float]
    bounds: SVIParams


class FilterConfig(BaseModel):
    n_particles: int
    resampling_threshold: float
    process_noise: List[float]
    observation_noise: float


class DataConfig(BaseModel):
    noise_level: float
    n_points: int
    strike_range: List[float]


class SimConfig(BaseModel):
    true_params: List[float]
    n_ticks: int
    noise_level: float
    strike_range: List[float]
    n_points: int


class Config(BaseModel):
    # Allows the model to accept extra fields without crashing
    model_config = ConfigDict(extra="ignore")

    model: ModelConfig
    sim: SimConfig
    data: DataConfig
    filter: Optional[FilterConfig] = None


def load_config(path: str) -> Config:
    """Loads and validates the YAML configuration."""
    try:
        with open(path, "r") as f:
            raw_data = yaml.safe_load(f)
        return Config(**raw_data)
    except Exception as e:
        # Professional logging of config errors
        print(f"CRITICAL: Failed to load config at {path}. Error: {e}")
        raise
