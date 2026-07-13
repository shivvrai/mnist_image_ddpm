from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    digit: int = Field(..., ge=0, le=10, description="Digit to generate (0-9) or 10 for unconditional")
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0, description="Classifier-free guidance scale")
    seed: int = Field(42, description="Random seed for generation")
    model_id: str = Field("mnist-ddpm", description="Model ID to use for generation")

class GenerateFromSketchRequest(GenerateRequest):
    sketch_b64: str = Field(..., description="Base64-encoded PNG data URL of the sketch")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Denoising strength: 0.0=no change, 1.0=full noise (ignores sketch)")

class GenerateResponse(BaseModel):
    image_b64: str
    digit: int
    guidance_scale: float
    seed: int
    generation_time_ms: int
    model_name: str
    predicted_digit: Optional[int] = None
class ClassifySketchRequest(BaseModel):
    sketch_b64: str = Field(..., description="Base64-encoded PNG data URL of the sketch")

class ClassifySketchResponse(BaseModel):
    predicted_digit: Optional[int] = None
    confidence: Optional[float] = None
