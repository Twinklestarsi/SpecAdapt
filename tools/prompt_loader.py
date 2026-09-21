#!/usr/bin/env python3
"""
Simple Prompt Loader for Design Analysis Workflow

This module provides functionality to load agent prompts from a structured JSON file.
No complex parsing needed - just direct JSON loading.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    description: str
    model: str
    temperature: float
    prompt: str


class PromptLoader:
    """Simple prompt loader that reads from JSON file"""

    def __init__(self, prompt_file_path: str = ""):
        """
        Initialize the prompt loader

        Args:
            prompt_file_path: Path to the prompts JSON file
        """
        self.prompt_file_path = prompt_file_path
        self.agents: Dict[str, AgentConfig] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompts from the JSON file"""
        if not self.prompt_file_path:
            # Default to prompts file in the same directory
            self.prompt_file_path = str(Path(__file__).parent / "agent_prompts.json")

        if not os.path.exists(self.prompt_file_path):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file_path}")

        try:
            with open(self.prompt_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load metadata
            self.metadata = data.get('metadata', {})

            # Load agent configurations
            agents_data = data.get('agents', {})
            for agent_id, agent_info in agents_data.items():
                self.agents[agent_id] = AgentConfig(
                    name=agent_info.get('name', agent_id),
                    description=agent_info.get('description', ''),
                    model=agent_info.get('model', 'gpt-4'),
                    temperature=agent_info.get('temperature', 0.3),
                    prompt=agent_info.get('prompt', '')
                )

            print(f"Loaded {len(self.agents)} agent configurations from {self.prompt_file_path}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompt file {self.prompt_file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompts from {self.prompt_file_path}: {e}")

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get configuration for a specific agent

        Args:
            agent_name: Name of the agent (e.g., "clock_information")

        Returns:
            AgentConfig object or None if not found
        """
        # Normalize agent name (remove common suffixes)
        normalized_name = agent_name.lower().replace(' ', '_')
        if normalized_name.endswith('_agent'):
            normalized_name = normalized_name[:-6]  # Remove '_agent' suffix
        if normalized_name.endswith('_extraction_agent'):
            normalized_name = normalized_name[:-17]  # Remove '_extraction_agent' suffix

        return self.agents.get(normalized_name)

    def get_prompt(self, agent_name: str, **kwargs) -> str:
        """
        Get a formatted prompt for a specific agent

        Args:
            agent_name: Name of the agent
            **kwargs: Variables to substitute in the prompt

        Returns:
            Formatted prompt string
        """
        config = self.get_agent_config(agent_name)
        if not config:
            raise ValueError(f"Agent configuration not found for: {agent_name}")

        try:
            formatted_prompt = config.prompt.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Missing variable {e} in prompt for {agent_name}")
            formatted_prompt = config.prompt
        except Exception as e:
            print(f"Error formatting prompt for {agent_name}: {e}")
            formatted_prompt = config.prompt

        return formatted_prompt

    def get_model_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific agent

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with model and temperature settings
        """
        config = self.get_agent_config(agent_name)
        if not config:
            return {"model": "gpt-4", "temperature": 0.3}

        return {
            "model": config.model,
            "temperature": config.temperature
        }

    def list_agents(self) -> Dict[str, str]:
        """
        List all available agents with their descriptions

        Returns:
            Dictionary mapping agent names to descriptions
        """
        return {name: config.description for name, config in self.agents.items()}

    def reload_prompts(self):
        """Reload prompts from the file"""
        self.agents.clear()
        self.metadata.clear()
        self._load_prompts()

    def validate_prompt_variables(self, agent_name: str, **kwargs) -> bool:
        """
        Validate that all required variables are provided for a prompt

        Args:
            agent_name: Name of the agent
            **kwargs: Variables to check

        Returns:
            True if all variables are present, False otherwise
        """
        config = self.get_agent_config(agent_name)
        if not config:
            return False

        # Find all {variable} placeholders in the prompt
        import re
        variables = re.findall(r'\{([^}]+)\}', config.prompt)

        missing_vars = [var for var in variables if var not in kwargs]

        if missing_vars:
            print(f"Missing variables for {agent_name}: {missing_vars}")
            return False

        return True


# Global instance for easy access
_global_prompt_loader = None


def get_prompt_loader(prompt_file_path: str = "") -> PromptLoader:
    """
    Get the global prompt loader instance

    Args:
        prompt_file_path: Path to prompts file (only used on first call)

    Returns:
        PromptLoader instance
    """
    global _global_prompt_loader
    if _global_prompt_loader is None:
        _global_prompt_loader = PromptLoader(prompt_file_path)
    return _global_prompt_loader


def get_agent_prompt(agent_name: str, **kwargs) -> str:
    """
    Convenience function to get a formatted prompt for an agent

    Args:
        agent_name: Name of the agent
        **kwargs: Variables to substitute

    Returns:
        Formatted prompt string
    """
    loader = get_prompt_loader()
    return loader.get_prompt(agent_name, **kwargs)


def get_agent_model_config(agent_name: str) -> Dict[str, Any]:
    """
    Convenience function to get model configuration for an agent

    Args:
        agent_name: Name of the agent

    Returns:
        Dictionary with model and temperature settings
    """
    loader = get_prompt_loader()
    return loader.get_model_config(agent_name)


if __name__ == "__main__":
    # Test the prompt loader
    try:
        loader = PromptLoader()

        print("Available agents:")
        for name, description in loader.list_agents().items():
            model_config = loader.get_model_config(name)
            print(f"  - {name}: {description}")
            print(f"    Model: {model_config['model']}, Temp: {model_config['temperature']}")

        # Test getting a prompt with variables
        test_prompt = loader.get_prompt(
            "clock_information",
            crg_content="// Example CRG content\nmodule crg(...);"
        )
        print(f"\nSample clock prompt preview (first 200 chars):")
        print(test_prompt[:200] + "...")

    except Exception as e:
        print(f"Error testing prompt loader: {e}")