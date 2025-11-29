# backend/services/llm_service.py
import logging
import os
import requests
import json
from typing import Optional, List, Dict, Any
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# Try to keep compatibility with langchain_openai if installed
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except Exception:
    ChatOpenAI = None
    LANGCHAIN_AVAILABLE = False


class LocalChat:
    """
    Wrapper for a local LLM HTTP API (runllamafile / llama.cpp server via ngrok).
    Uses the robust extraction approach from the test script.
    Exposes:
      - .predict(prompt: str) -> str
      - .generate(messages=[...]) -> object with .generations shape expected by crewai/langchain
      - .bind(**kwargs) -> self   (crewai expects this)
      - __call__(prompt) -> str   (convenience)
    """

    def __init__(self, host: str = "https://022de468e7e2.ngrok-free.app/v1", model: str = None,
                 temperature: float = 0.3, system_prompt: Optional[str] = None,
                 max_tokens: int = 1024):
        self.host = host.rstrip('/')
        self.model = model or self._get_model_id()
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._bound_kwargs: Dict[str, Any] = {}
        
        # Headers for ngrok
        self.headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "yes",
        }

    def _get_model_id(self) -> str:
        """Fetch the model ID from the server's /models endpoint"""
        try:
            r = requests.get(
                f"{self.host}/models", 
                headers=self.headers, 
                timeout=10
            )
            j = r.json()
            for item in j.get("data", []):
                if item.get("id"):
                    logger.info(f"Auto-detected model: {item['id']}")
                    return item["id"]
        except Exception as e:
            logger.warning(f"Could not fetch model ID: {e}")
        
        # Fallback
        return "llama.gguf"

    def _extract_choice_text(self, resp) -> Optional[str]:
        """
        Robust text extraction from LLM response.
        Handles both dict-style (requests) and object-style responses.
        """
        # Dict-style (requests.json())
        if isinstance(resp, dict):
            try:
                choices = resp.get("choices", [])
                if choices:
                    c0 = choices[0]
                    if isinstance(c0, dict):
                        m = c0.get("message", {})
                        if isinstance(m, dict) and m.get("content"):
                            return m.get("content")
                        if c0.get("text"):
                            return c0.get("text")
                    return json.dumps(c0, default=str)
            except Exception:
                pass

            # Top-level fallback
            for k in ("text", "output", "message", "content"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
            return None

        # Object-style (for OpenAI client compatibility)
        try:
            choices = getattr(resp, "choices", None)
            if choices:
                c0 = choices[0]
                msg = getattr(c0, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content:
                        return content
                    try:
                        v = vars(msg)
                        if "content" in v:
                            return v["content"]
                    except Exception:
                        pass
                if hasattr(c0, "text"):
                    return getattr(c0, "text")
                return str(c0)
        except Exception:
            pass

        return None

    # -------------------
    # Compatibility API
    # -------------------
    def bind(self, **kwargs):
        """
        Bind parameters (e.g. stop tokens) similar to other LLM wrappers.
        crewai calls `.bind(stop=[...])` — we store it and return self.
        """
        logger.debug(f"LocalChat.bind called with {kwargs}")
        self._bound_kwargs.update(kwargs or {})
        return self

    def generate(self, messages: Optional[List[Dict[str, str]]] = None, 
                prompt: Optional[str] = None, **kwargs):
        """
        Generate response given chat messages or a prompt.
        Accepts OpenAI-style messages: [{"role":"user","content":"..."}, ...]
        Returns a SimpleNamespace with .generations = [[SimpleNamespace(text=...)]]
        which is compatible with crewai/langchain minimal expectations.
        """
        # Build the prompt
        if messages:
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"[{role}]: {content}")
            prompt_text = "\n".join(parts)
        elif prompt:
            prompt_text = prompt
        else:
            raise ValueError("generate requires 'messages' or 'prompt'")

        # Include system prompt if present
        if self.system_prompt:
            prompt_text = f"[system]: {self.system_prompt}\n\n" + prompt_text

        logger.debug(f"LocalChat.generate prompt length={len(prompt_text)}")

        # Use predict to call the server
        text = self.predict(prompt_text, **kwargs)

        # Wrap into expected shape
        gen = SimpleNamespace(text=text)
        return SimpleNamespace(generations=[[gen]])

    def predict(self, prompt: str, **kwargs) -> str:
        """
        Synchronous method that calls the local /v1/chat/completions.
        Uses the robust extraction approach from the test script.
        Includes retry logic and better timeout handling.
        """
        url = f"{self.host}/chat/completions"
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)
        timeout = kwargs.get("timeout", 120)  # Increased default timeout
        max_retries = kwargs.get("max_retries", 2)

        # Build messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Include any bound kwargs (e.g. stop)
        for k, v in self._bound_kwargs.items():
            if k not in body:
                body[k] = v

        # Retry logic
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.debug(f"POST {url} model={model} temp={temperature} (attempt {attempt + 1}/{max_retries})")
                
                # Make request with timeout
                resp = requests.post(
                    url, 
                    headers=self.headers, 
                    json=body, 
                    timeout=timeout
                )
                
                resp.raise_for_status()
                
                # Parse response
                try:
                    data = resp.json()
                except Exception:
                    logger.warning(f"Non-JSON response: {resp.text[:400]}")
                    return resp.text.strip()
                
                # Extract text using robust method
                text = self._extract_choice_text(data)
                
                if text:
                    return text.strip()
                
                # Final fallback
                logger.warning(f"Could not extract text from response: {json.dumps(data)[:500]}")
                return resp.text.strip()
                
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"LLM request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with longer timeout...")
                    timeout = timeout * 1.5  # Increase timeout for retry
                    continue
                
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.error(f"Connection error to LLM server: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying connection...")
                    continue
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.exception(f"LocalChat request error: {e}")
                if attempt < max_retries - 1:
                    continue
                
            except Exception as e:
                last_error = e
                logger.exception(f"LocalChat predict error: {e}")
                if attempt < max_retries - 1:
                    continue
        
        # All retries failed
        error_msg = f"LLM request failed after {max_retries} attempts: {str(last_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def __call__(self, prompt: str, **kwargs) -> str:
        """Convenience alias for predict"""
        return self.predict(prompt, **kwargs)


class LLMService:
    """
    Factory for chat LLMs. Choose local Llama server by setting
    DEFAULT_LLM_MODEL=llama-local or similar in env / config.
    """
    def __init__(self):
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "llama-local").lower()
        
        # Local host/port config (override via env)
        # IMPORTANT: Set this to your ngrok URL
        self.local_host = os.getenv(
            "LLM_HOST", 
            "https://ff0afe61b1bf.ngrok-free.app/v1"
        )
        self.local_model_name = os.getenv("LLM_MODEL_NAME", None)  # Auto-detect if None
        self.default_temperature = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.3"))

    def get_chat_llm(self, model: Optional[str] = None, 
                    temperature: float = None, 
                    system_prompt: Optional[str] = None):
        """
        Return an object with .predict(prompt) -> str, .generate(messages=...), .bind(...)
        If model or DEFAULT_LLM_MODEL contains 'llama' or 'local', use LocalChat.
        Otherwise try to return a ChatOpenAI (langchain) wrapper.
        """
        temperature = self.default_temperature if temperature is None else temperature
        model_name = (model or self.default_model).lower()

        if "llama" in model_name or "local" in model_name or "gguf" in model_name:
            logger.info(f"Using LocalChat at {self.local_host}")
            return LocalChat(
                host=self.local_host, 
                model=self.local_model_name, 
                temperature=temperature, 
                system_prompt=system_prompt
            )

        if LANGCHAIN_AVAILABLE:
            try:
                logger.info(f"Using ChatOpenAI model={model_name}")
                return ChatOpenAI(model=model_name, temperature=temperature)
            except Exception as e:
                logger.warning(f"Failed to create ChatOpenAI: {e}")

        raise RuntimeError(
            "No suitable LLM backend available. "
            "Set DEFAULT_LLM_MODEL to 'llama-local' or install langchain_openai."
        )

    def predict(self, prompt: str, model: Optional[str] = None, 
               temperature: float = None) -> str:
        """Convenience method to get an LLM and predict in one call"""
        llm = self.get_chat_llm(model=model, temperature=temperature)
        return llm.predict(prompt)