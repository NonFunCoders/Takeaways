from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import logging
import asyncio
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class CodeResponse(BaseModel):
    code: str
    explanation: Optional[str] = None

app = FastAPI(
    title="Takeaways API",
    description="API for the Takeaways coding assistant",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelService:
    def __init__(self, model_path: str):
        """Initialize the model service.
        
        Args:
            model_path: Path to the trained model
        """
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Generate code from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the output
            
        Returns:
            Generated code
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        if stream:
            # Stream tokens one at a time
            generated_ids = []
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=1,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1
                    )
                    
                next_token = outputs[0][-1].unsqueeze(0)
                generated_ids.append(next_token)
                
                # Yield current output
                current_output = self.tokenizer.decode(
                    torch.cat(generated_ids),
                    skip_special_tokens=True
                )
                yield current_output
                
                # Check for end of generation
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # Update input for next token
                inputs = {"input_ids": outputs}
        else:
            # Generate complete response at once
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize model service
model_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model service on startup."""
    global model_service
    model_path = Path(__file__).parent.parent / "exported"
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. "
            "Please export the model first using the trainer."
        )
    model_service = ModelService(str(model_path))

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate code from prompt."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    try:
        response = await model_service.generate(
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            stream=False
        )
        
        # Split response into code and explanation
        parts = response.split("### Response:\n")
        if len(parts) > 1:
            code = parts[1].strip()
            explanation = None
            
            # Check for explanation section
            if "### Explanation:" in code:
                code_parts = code.split("### Explanation:")
                code = code_parts[0].strip()
                explanation = code_parts[1].strip() if len(code_parts) > 1 else None
                
            return CodeResponse(code=code, explanation=explanation)
        else:
            return CodeResponse(code=response)
            
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """Stream code generation over WebSocket."""
    await websocket.accept()
    
    if not model_service:
        await websocket.close(code=1011, reason="Model not initialized")
        return
        
    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            request = CodeRequest.parse_raw(data)
            
            # Stream generation
            async for output in model_service.generate(
                request.prompt,
                request.max_new_tokens,
                request.temperature,
                stream=True
            ):
                await websocket.send_text(json.dumps({"code": output}))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
