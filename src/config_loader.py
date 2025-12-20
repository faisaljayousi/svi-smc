from typing import List

import yaml
from pydantic import BaseModel, field_validator


class SVIParams(BaseModel):
    """Bounds for SVI parameters: [a, b, rho, m, sigma]"""

    lower: List[float]
    upper: List[float]

    @field_validator("lower", "upper")
    @classmethod
    def check_length(cls, v):
        if len(v) != 5:
            raise ValueError("SVI bounds must have exactly 5 elements")
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


class Config(BaseModel):
    model: ModelConfig
    filter: FilterConfig
    data: DataConfig


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
