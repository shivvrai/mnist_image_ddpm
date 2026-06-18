from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    digit: int = Field(..., ge=0, le=9, description="Digit to generate (0-9)")
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0, description="Classifier-free guidance scale")
    seed: int = Field(42, description="Random seed for generation")
    model_id: str = Field("mnist-ddpm", description="Model ID to use for generation")

class GenerateResponse(BaseModel):
    image_b64: str
    digit: int
    guidance_scale: float
    seed: int
    generation_time_ms: int
    model_name: str
