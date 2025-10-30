import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import sys
from pathlib import Path
from anthropic import Anthropic
import openai
import weave

BASE_DIR = Path(__file__).parent.parent


sys.path.insert(0, str(BASE_DIR / "src"))
from trace_parser import TraceParser
from failure_classes import (
    MAKE_FAILURE_PROMPTS,
    PARSE_RESPONSE_FUNCTIONS,
    parse_failure_definitions2,
)

try:
    from failure_classifier_mode22 import ENHANCED_PROMPT_MAKERS

    # Merge enhanced prompts
    MAKE_FAILURE_PROMPTS.update(ENHANCED_PROMPT_MAKERS)
except ImportError:
    pass  # Enhanced Mode 2.2 prompts not available

try:
    from failure_classifier_mode22_correct import CORRECTED_PROMPT_MAKERS

    # Merge corrected Mode 2.2 prompts
    MAKE_FAILURE_PROMPTS.update(CORRECTED_PROMPT_MAKERS)
except ImportError:
    pass  # Corrected Mode 2.2 prompts not available

load_dotenv()


class FailureProcessor:
    """
    Base class for processing failure traces with shared configuration and output format.

    Args:
        config: Dictionary containing configuration parameters
        max_trace_length: Maximum trace length to process


    All FailureProcessor subclasses must output failure modes according to OUTPUT_MODE_RUBRIC:

    the base class output is
    - category: string (failure classification category)
    - evidence: string (specific evidence from trace).
    - confidence: float 0.0-1.0 (confidence score)
    - required_skill: string (skill needed to avoid this failure)
    """

    def __init__(self, config=None):
        """
        Initialize the failure processor with shared configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.max_trace_length = self.config.get("max_trace_length", None)

    def process_trace(
        self, trace: str, task_description: str = "", max_trace_length: int = None
    ) -> Dict:
        """
        Process a failure trace and return standardized output.

        Args:
            trace: The trace text to process
            task_description: Optional description of what the task was supposed to do

        Returns:
            Standardized result dictionary with metadata
        """
        raise NotImplementedError("Subclasses must implement process_trace method")

    def _prepare_text(self, trace: str, task_description: str) -> str:
        """Prepare text which will be fed to the model.
        Here we combine the task description and the trace.
        """

        # Truncate trace if too long and max_trace_length is set
        if self.max_trace_length and len(trace) > self.max_trace_length:
            trace = trace[: self.max_trace_length] + "... [truncated]"

        # Combine task description and trace
        if task_description:
            text = f"Task: {task_description}\n\nExecution Trace:\n{trace}"
        else:
            text = f"Execution Trace:\n{trace}"

        return text

    def _load_system_message(self, system_message_id: str) -> Optional[str]:
        """Load system message text from prompts/system_messages/{id}.txt; return None on failure."""
        try:
            system_path = (
                BASE_DIR / "prompts" / "system_messages" / f"{system_message_id}.txt"
            )
            if system_path.exists():
                return system_path.read_text()
        except:
            raise ValueError(f"System message {system_message_id} not found")


class FailureAnalyzerJudge(FailureProcessor):
    """Analyze failure traces using LLM classification as judge."""

    def __init__(self, config=None):
        """
        Initialize the failure analyzer.

        Args:
            config: Configuration dictionary with keys like 'llm_provider', 'model_name', etc.
        """
        super().__init__(config)
        self.model_name = self.config.get(
            "model_name", None
        )  # Will be set based on provider
        self.llm_provider = self.config.get("llm_provider", self._get_model_provider())
        self.failure_prompt_type = self.config.get("failure_prompt_type", "mast")
        # Check if Weave should be explicitly disabled (for cases where it's already initialized)
        self.skip_weave_init = self.config.get("skip_weave_init", True)
        # System message selection: 0 for JSON formatting, 1 for expert evaluator (default)
        self.system_message_id = self.config.get("system_message_id", "v2")

        if self.llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = self.config.get("model_name", "claude-3-haiku-20240307")
        elif self.llm_provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai
            self.model_name = self.config.get("model_name", "gpt-4o-mini")
        elif self.llm_provider == "local":
            # Local model
            self.client = None
            self.model_name = self.config.get("model_name", "microsoft/deberta-v3-base")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _get_model_provider(self) -> str:
        """Get the model provider."""
        if self.model_name in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
            "gpt-5-nano",
            "o1",
            "o1-mini",
            "o1-preview",
        ]:
            return "openai"
        elif self.model_name in [
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-1-20250805",
        ]:
            return "anthropic"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _build_taxonomy_appendix(self) -> str:
        """Appendix string with definitions, examples, and expected JSON schema."""
        try:
            # Use failure_prompt_type as taxonomy version directory (e.g., tb_v0)
            taxonomy_dir = BASE_DIR / "taxonomies" / self.failure_prompt_type
            definitions_path = taxonomy_dir / "definitions.txt"
            examples_path = taxonomy_dir / "examples.txt"
            definitions = definitions_path.read_text()
            examples = examples_path.read_text()
            schema = parse_failure_definitions2(definitions)
            appendix = (
                f"\n\n## Failure Mode Definitions\n{definitions}\n\n"
                f"## Examples\n{examples}\n\n"
                # f"## Expected JSON Output\n{json.dumps(schema, indent=2)}\n\n"
                # f"Follow the schema exactly. Use lowercase labels 'yes'|'no'|'unclear'."
            )
            return appendix
        except Exception:
            return ""

    def _get_response(self, prompt: str, retry_count: int = 0) -> Dict[str, any]:
        """Get the response JSON from the LLM with retry logic and JSON validation.

        Args:
            prompt: The prompt to send to the LLM
            retry_count: Current retry attempt number

        Returns:
            The LLM's response text
        """
        # Define system messages once for both providers
        system_msg = None
        if self.failure_prompt_type in ["tb0", "mast", "base", "tb1", "tb2"]:
            system_msg = self._load_system_message(self.system_message_id)
            # If using tb2, append taxonomy/schema to whichever system message was selected
            if self.failure_prompt_type == "tb2":
                base_msg = system_msg or ""
                system_msg = base_msg + self._build_taxonomy_appendix()

        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                temperature=0.0
                if retry_count == 0
                else 0.1,  # Slightly increase temp on retry
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
                system=system_msg if system_msg else None,
            )
            return response.content[0].text

        elif self.llm_provider == "openai":
            msg = [{"role": "user", "content": prompt}]
            model = self.model_name.lower()

            if system_msg:
                msg.insert(0, {"role": "system", "content": system_msg})

            kwargs = {"model": self.model_name, "messages": msg, "timeout": 300}

            if "o1" in model:
                kwargs.update(
                    {
                        "reasoning": {"effort": "low"},
                        "text": {"verbosity": "low"},
                        "seed": 42,
                    }
                )
            elif "gpt-5" in model:
                # GPT-5 uses default parameters only
                # Try to use JSON mode if available
                if self.failure_prompt_type in ["tb0", "mast", "base"]:
                    kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs.update(
                    {
                        "temperature": 0.0 if retry_count == 0 else 0.1,
                        "top_p": 1.0,
                        # "max_tokens": ,
                    }
                )
                # Try to use JSON mode for GPT-4 models
                if "gpt-4" in model and self.failure_prompt_type in [
                    "tb0",
                    "mast",
                    "base",
                    "tb1",
                    "tb1_refined",
                ]:
                    kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        elif self.llm_provider == "local":
            return self._local_classify(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_response(self, response: str):
        """Parse the response from the LLM.

        Returns:
            For mast: Tuple of (failure_modes, full_analysis)
            For others: Just failure_modes dict
        """
        parse_response_function = PARSE_RESPONSE_FUNCTIONS[self.failure_prompt_type]
        try:
            return parse_response_function(response)
        except Exception as e:
            print(f"ERROR parsing {self.failure_prompt_type} response: {e}")
            print(f"Response preview (first 500 chars): {response[:500]}")
            # Log full response to a debug file if it's a JSON issue
            if "json" in str(e).lower() or "JSON" in str(e):
                debug_file = f"/tmp/json_parse_error_{self.failure_prompt_type}.txt"
                with open(debug_file, "w") as f:
                    f.write(f"Error: {e}\n\n")
                    f.write(f"Full response:\n{response}")
                print(f"Full response saved to {debug_file} for debugging")
            raise

    @weave.op()
    def process_trace(
        self,
        trace: str,
        task_description: str = "",
        trial_id: str = None,
        reward: Optional[str] = None,
        verifier_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Classify a failure trace into one or more failure modes using LLM.

        Args:
            trace: The trace text to classify
            task_description: Optional description of what the task was supposed to do
            trial_id: Optional trial ID for tracking
            reward: Optional reward score from verifier (for tb0 prompt)
            verifier_data: Optional verifier data dict (for tb0 prompt)

        Returns:
            For mast: Tuple of (failure_modes, full_analysis)
            For others: Just failure_modes dict
        """
        # Build the classification prompt
        prompt = self._prepare_text(
            trace=trace,
            task_description=task_description,
            reward=reward,
            verifier_data=verifier_data,
        )

        # Try up to 3 times with validation
        max_retries = 3
        last_error = None

        for retry in range(max_retries):
            try:
                response_json = self._get_response(prompt, retry_count=retry)

                # Quick validation before parsing
                if response_json and isinstance(response_json, str):
                    cleaned = response_json.strip()
                    # Remove any markdown wrappers
                    if "```json" in cleaned.lower():
                        parts = cleaned.split("```json", 1)
                        if len(parts) > 1:
                            cleaned = parts[1].split("```")[0]
                    elif "```" in cleaned:
                        parts = cleaned.split("```")
                        if len(parts) >= 3:
                            cleaned = parts[1]

                    cleaned = cleaned.strip()

                    # Check if it looks like JSON
                    if not (cleaned.startswith("{") or cleaned.startswith("[")):
                        # Maybe there's text before the JSON?
                        json_start = cleaned.find("{")
                        if json_start > 0:
                            cleaned = cleaned[json_start:]
                        else:
                            raise ValueError(
                                f"Response does not appear to be JSON: {cleaned[:100]}..."
                            )

                    response_json = cleaned

                # Try to parse the response
                result = self._parse_response(response_json)

                # Basic validation of the result
                if self.failure_prompt_type in [
                    "tb0",
                    "tb1",
                    "tb1_refined",
                    "tb1_mode22",
                    "tb1_mode22_correct",
                    "tb2",
                    "mast",
                ] and isinstance(result, tuple):
                    # For tb0/tb1/mast, should return (failure_modes_dict, analysis_str)
                    if len(result) == 2 and isinstance(result[0], dict):
                        return result
                elif isinstance(result, dict):
                    return result

                raise ValueError(
                    f"Invalid result structure from parser: {type(result)}"
                )

            except Exception as e:
                last_error = e
                if retry < max_retries - 1:
                    print(
                        f"Retry {retry + 1}/{max_retries} for trial {trial_id} due to: {str(e)[:200]}"
                    )

                    # Modify prompt to emphasize JSON output on retry
                    if retry == 0:
                        prompt = prompt.replace(
                            "Now output the JSON response",
                            "CRITICAL: Return ONLY valid JSON with double quotes. No Python dicts. Now output the JSON response",
                        )
                    elif retry == 1:
                        prompt = prompt.replace(
                            "CRITICAL: Return ONLY valid JSON",
                            "FINAL ATTEMPT: You MUST return valid JSON starting with { and using double quotes throughout",
                        )
                else:
                    print(
                        f"Failed after {max_retries} attempts for trial {trial_id}: {e}"
                    )

        # If all retries failed, raise the last error
        if last_error:
            raise last_error

        # Shouldn't reach here, but return empty result if it does
        return (
            ({}, "")
            if self.failure_prompt_type in ["mast", "tb0", "tb1", "tb1_refined"]
            else {}
        )

    def _prepare_text(
        self,
        trace: str,
        task_description: Optional[str] = None,
        reward: Optional[str] = None,
        verifier_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the prompt for LLM classification."""
        # Check trace length
        if self.max_trace_length and len(trace) > self.max_trace_length:
            strict_mode = self.config.get("strict_length_check", False)
            if strict_mode:
                raise ValueError(
                    f"Execution trace is too long ({len(trace)} characters). "
                    f"Maximum allowed length is {self.max_trace_length} characters."
                )

        # Pass additional parameters for tb0, tb1, and enhanced prompts
        if self.failure_prompt_type in [
            "tb0",
            "tb1",
            "tb1_refined",
            "tb1_mode22",
            "tb1_mode22_correct",
            "tb2",
        ]:
            prompt = MAKE_FAILURE_PROMPTS[self.failure_prompt_type](
                trace, task_description, reward=reward, verifier_data=verifier_data
            )
        else:
            prompt = MAKE_FAILURE_PROMPTS[self.failure_prompt_type](
                trace, task_description
            )

        return prompt


class FailureEmbedder(FailureProcessor):
    """Generate embeddings for failure traces using specified embedding distance and model."""

    def __init__(self, config=None):
        """
        Initialize the failure embedder.

        Args:
            config: Configuration dictionary with keys like 'embedding_distance', 'model_name', etc.
        """
        super().__init__(config)
        self.embedding_distance = self.config.get("embedding_distance", "cosine")
        self.model_name = self.config.get("model_name", "text-embedding-3-small")

        # Initialize OpenAI client for embeddings
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @weave.op()
    def process_trace(self, trace: str, task_description: str = "") -> Dict:
        """
        Process a failure trace and return embedding results in standardized format.

        Args:
            trace: The trace text to embed
            task_description: Optional description of what the task was supposed to do

        Returns:
            Standardized result dictionary with embedding vector
        """
        embedding = self.embed_trace(trace, task_description)

        metadata = {
            "embedding_distance": self.embedding_distance,
            "model_name": self.model_name,
            "embedding_dimension": len(embedding),
            "trace_length": len(trace),
            "task_description": task_description,
        }

        return self._create_standardized_output("embedding", embedding, metadata)

    def embed_trace(self, trace: str, task_description: str = "") -> List[float]:
        """
        Generate embeddings for a failure trace.

        Args:
            trace: The trace text to embed
            task_description: Optional description of what the task was supposed to do

        Returns:
            List of embedding values (floats)
        """
        # Prepare the text for embedding using parent method
        text_to_embed = self._prepare_text(trace, task_description)

        try:
            response = self.client.embeddings.create(
                model=self.model_name, input=text_to_embed, encoding_format="float"
            )
            return response.data[0].embedding

        except Exception as e:
            print(f"Error generating embedding: {e}", flush=True)
            # Return zero vector on error
            return [0.0] * 1536  # Default size for text-embedding-3-small

    def compute_distance(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute distance between two embeddings using the specified distance metric.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Distance value (float)
        """
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        if self.embedding_distance == "cosine":
            # Cosine similarity, converted to distance
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            cosine_sim = dot_product / (norm1 * norm2)
            return 1 - cosine_sim
        elif self.embedding_distance == "euclidean":
            return np.linalg.norm(vec1 - vec2)
        elif self.embedding_distance == "manhattan":
            return np.sum(np.abs(vec1 - vec2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.embedding_distance}")


FAILURE_CLASSIFIERS = {"judge": FailureAnalyzerJudge, "embedder": FailureEmbedder}


def get_config(
    classifier_type: str,
    failure_prompt_type: str = "mast",
    model_name: str = None,
    system_message_id: str = "v1",
    max_trace_length: int = None,
):
    """Get the configuration for a classifier.

    Args:
        classifier_type: Type of classifier ("judge" or "embedder")
        failure_prompt_type: Prompt type ("base", "mast", "timeout", "tb0", "tb1", "tb1_refined")
        model_name: Name of the model to use
        system_message_id: Which system message to use ("v0"=JSON formatting, "v1"=expert evaluator, "v2"=TB1 comprehensive)
    """
    if classifier_type == "judge":
        config = {
            "model_name": model_name,
            "failure_prompt_type": failure_prompt_type,
            "system_message_id": system_message_id,
        }
        return config
    elif classifier_type == "embedder":
        return {"embedding_distance": "cosine", "model_name": "text-embedding-3-small"}
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")


if __name__ == "__main__":
    # Initialize Weave with project name
    debug = False
    if not debug:
        weave_project = os.getenv("WANDB_PROJECT", "judge")
        weave.init(weave_project)

    # Example usage
    parser = TraceParser(BASE_DIR / "traces")

    trial_id = (
        "0b1249b7-84df-4604-95c0-94a8ac7f2cad"  # Large trace - may take time with GPT-5
    )
    failure_classifier = "judge"
    model_name = "gpt-5"
    prompt_type = (
        "tb2"  # Can be "base", "mast", "timeout", "tb0", "tb1", or "tb1_refined"
    )
    system_message_id = "v4"

    # Read the trace
    trace = parser.parse_trace(trial_id)
    trace_text = trace.to_json(include_metadata=False)
    reward = trace.get_reward()
    verifier_data = trace.verifier_data

    # write to json files
    if debug:
        # write prompt to file: `$TBENCH/prompts/failure_classifier_prompt.txt`
        prompt_path = (
            BASE_DIR / "prompts" / "failure_classifier_trace" / f"{trial_id}.txt"
        )
        prompt_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
        with open(prompt_path, "w") as f:
            f.write(trace_text)

    classifier = FAILURE_CLASSIFIERS[failure_classifier](
        config=get_config(
            classifier_type="judge",
            failure_prompt_type=prompt_type,
            model_name=model_name,
            system_message_id=system_message_id,
            max_trace_length=None,
        )
    )

    if debug:
        # Print classification prompt:
        prompt = classifier._prepare_text(
            trace=trace_text, reward=reward, verifier_data=verifier_data
        )

        # write prompt to file: `$TBENCH/prompts/failure_classifier_prompt.txt`
        prompt_path = (
            BASE_DIR
            / "prompts"
            / f"failure_classifier_prompt_{prompt_type}"
            / f"{trial_id}.txt"
        )
        prompt_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
        with open(prompt_path, "w") as f:
            f.write(prompt)

        print(f"Generated {prompt_type} prompt for trial {trial_id}")
        print(f"Prompt saved to: {prompt_path}")
        print(f"\n \nPrompt : \n\n{prompt}")
        exit()

    # Test classification:

    print(f"Processing trial {trial_id} with {model_name}...")
    print(f"Trace size: {len(trace_text)} characters")

    failure_modes = classifier.process_trace(
        trace=trace_text, trial_id=trial_id, reward=reward, verifier_data=verifier_data
    )
    print("\nResults:")
    print(failure_modes, flush=True)
