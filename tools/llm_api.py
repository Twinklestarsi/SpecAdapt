# coding:utf-8
# llm生成dc脚本配置文件

import os
import traceback
import base64
import logging
from pathlib import Path
from typing import Union, List, Dict, Any
from openai import OpenAI



_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=(
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
            ),
        )
    return _client

def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        raise ValueError(f"Error encoding image {image_path}: {e}")


def create_multimodal_content(text: str, images: List[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Create multimodal content for OpenAI API with text and images.
    
    Args:
        text: Text content
        images: List of image file paths (optional)
        
    Returns:
        List of content objects for OpenAI API
    """
    content = []
    
    # Add text content
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    # Add image content
    if images:
        for image_path in images:
            # Determine image format from file extension
            image_path = Path(image_path)
            image_format = image_path.suffix.lower().lstrip('.')
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            elif image_format in ['gif', 'webp']:
                image_format = image_format
            else:
                # Default to jpeg for unknown formats
                image_format = 'jpeg'
                
            encoded_image = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{encoded_image}",
                    "detail": "high"  # Can be "low", "high", or "auto"
                }
            })
    
    return content


def call_llm(prompt: str, model: str = 'gpt-4o', temperature: float = 1.0,
             images: List[Union[str, Path]] = None, timeout=30) -> str:
    """
    Call LLM with support for both text and image inputs.

    Args:
        prompt: Text prompt
        model: Model name (should support vision for image inputs)
        temperature: Temperature for generation
        images: List of image file paths (optional)

    Returns:
        Response from the LLM
    """
    try:
        # Create multimodal content if images are provided
        if images:
            user_content = create_multimodal_content(prompt, images)
        else:
            user_content = prompt

        if "embedding" in model:
            response = _get_client().embeddings.create(
                model=model,
                input=user_content,
                # temperature=temperature,
                #max_tokens=8192,
            )
            # print("Pure response", response)
            # logging.info(response)
            res = response.data[0].embedding

        else:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                    "content": "You are a helpful AI agent"},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                #max_tokens=8192,
            )
            # print(response)
            # logging.info(response)
            res = response.choices[0].message.content
        return res
    except Exception as err:
        return f'API error: {err}'


if __name__ == "__main__":
    temperature = 1.0
    content = "鲁迅为什么会打周树人？"
    response = call_llm(content, model="gemini-2.5-pro", temperature=temperature, images=None)
    # response = call_llm(content, model="text-embedding-3-large", temperature=temperature, images=None)
    print("Text-only response:", response)
    print("="*128)

    
