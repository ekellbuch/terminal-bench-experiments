from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).parent.parent


def extract_dict_from_text(text: str) -> Optional[Dict]:
    """Extract dictionary from text that may be in Python dict or JSON format.

    This function handles cases where LLMs return Python dicts (single quotes)
    or JSON (double quotes), possibly with extra text.

    Args:
        text: Text that may contain a dictionary

    Returns:
        Parsed dictionary object or None if extraction fails
    """
    import ast

    # Clean up common markdown wrappers
    cleaned = text.strip()

    # Remove markdown code blocks
    if "```json" in cleaned.lower():
        parts = cleaned.split("```json", 1)
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0].strip()
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()

    # Method 1: Try JSON parsing first (fastest for valid JSON)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Method 2: Try ast.literal_eval for Python dict format (with single quotes)
    try:
        result = ast.literal_eval(cleaned)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Method 3: Find dictionary boundaries and extract
    start_idx = cleaned.find("{")
    if start_idx >= 0:
        # Find matching closing brace
        brace_count = 0
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == "{":
                brace_count += 1
            elif cleaned[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    dict_str = cleaned[start_idx : i + 1]

                    # Try ast.literal_eval first (handles Python dict format)
                    try:
                        result = ast.literal_eval(dict_str)
                        if isinstance(result, dict):
                            return result
                    except (ValueError, SyntaxError):
                        pass

                    # Try JSON parsing
                    try:
                        return json.loads(dict_str)
                    except json.JSONDecodeError:
                        pass

                    break

    # Method 4: Handle truncated responses by extracting what we can
    # Look for key-value pairs even if the dict is incomplete
    if "{" in cleaned:
        truncated_start = cleaned.find("{")
        truncated_dict = cleaned[truncated_start:]

        # Try to extract at least some key-value pairs
        summary_match = re.search(
            r"['\"]summary['\"]\s*:\s*['\"]([^'\"]*)['\"]", truncated_dict
        )
        task_match = re.search(
            r"['\"]task_completed['\"]\s*:\s*['\"]([^'\"]*)['\"]", truncated_dict
        )

        if summary_match or task_match:
            # Build a minimal dict from what we can extract
            extracted = {}
            if summary_match:
                extracted["summary"] = summary_match.group(1)
            if task_match:
                extracted["task_completed"] = task_match.group(1)

            # Add empty failure_modes if not found
            if "failure_modes" not in extracted:
                extracted["failure_modes"] = {}

            return extracted

    # If all methods fail, return None
    return None


FAILURE_MODE_RUBRIC = {
    "Instruction Misunderstanding": {
        "definition": "The agent misunderstood the natural language task.",
        "indicators": [
            "Executes irrelevant commands",
            "Solves only part of the task",
            "Attempts wrong subgoal"
        ],
        "examples": [
            "Task: copy file A to B -> Agent compresses file instead",
            "Task: run program -> Agent only lists files"
        ]
    },
    "Command Construction Error": {
        "definition": "The agent chose the correct tool but built the command incorrectly.",
        "indicators": [
            "Syntax errors (e.g., 'bash: syntax error')",
            "Invalid flags/options",
            "Misspelled commands ('cmakee' instead of 'cmake')"
        ],
        "examples": [
            "grep -xzf file.txt",
            "ls --not-a-real-flag"
        ]
    },
    "Environment / System Error": {
        "definition": "The command was valid but failed due to environment setup.",
        "indicators": [
            "Dependency missing ('command not found')",
            "File/path errors ('No such file or directory')",
            "Permission issues ('Permission denied')"
        ],
        "examples": [
            "python3: No module named requests",
            "ls: cannot access 'data/': No such file or directory"
        ]
    },
    "Execution Logic Error": {
        "definition": "Commands valid and environment correct, but reasoning flawed.",
        "indicators": [
            "Steps executed in wrong order",
            "Wrong outputs produced",
            "Redundant or looping actions"
        ],
        "examples": [
            "Runs 'make install' before 'make'",
            "Deletes source file before compilation"
        ]
    },
    "Memory / Context Failure": {
        "definition": "The agent lost track of its prior state or actions.",
        "indicators": [
            "Repeats same command with no progress",
            "Refers to nonexistent files",
            "Switches to wrong working directory"
        ],
        "examples": [
            "Creates results.txt but later runs 'cat output.txt'",
            "Repeats 'ls' endlessly"
        ]
    },
    "Error Recovery Failure": {
        "definition": "The agent failed to handle errors effectively.",
        "indicators": [
            "Ignores clear error messages",
            "Retries same failing command",
            "Stops after first failure without retry",
            "Marks task as completed even though it failed"
        ],
        "examples": [
            "command not found -> agent retries same line 3 times",
            "permission denied -> agent gives up"
        ]
    },
    "Timeout / Non-Completion": {
        "definition": "The agent did not complete the task.",
        "indicators": [
            "Trial exceeded max steps/tokens",
            "Agent hangs indefinitely",
            "No output until timeout"
        ],
        "examples": [
            "Agent outputs 'Thinking...' repeatedly until timeout",
            "Process killed due to step budget"
        ]
    }
}


# Output format specification for FailureProcessor classes
OUTPUT_MODE_RUBRIC = {
    "failure_modes": {
        "Instruction Misunderstanding": {
            "score": 0.0,
            "evidence": "",
            "required_skill": "Understand natural language task",
        },
        "Command Construction Error": {
            "score": 0.0,
            "evidence": "Error message: 'bash: syntax error'",
            "required_skill": "Build valid commands with correct syntax",
        },
        "Environment / System Error": {
            "score": 0.0,
            "evidence": "",
            "required_skill": "Check environment dependencies",
        },
        "Execution Logic Error": {
            "score": 0.0,
            "evidence": "",
            "required_skill": "Reason about correct execution order",
        },
        "Memory / Context Failure": {
            "score": 0.0,
            "evidence": "",
            "required_skill": "Maintain context across steps",
        },
        "Error Recovery Failure": {
            "score": 0.0,
            "evidence": "Agent ignored 'command not found' and stopped",
            "required_skill": "Handle errors and retry adaptively",
        },
        "Timeout / Non-Completion": {
            "score": 1.0,
            "evidence": "",
            "required_skill": "Complete within time budget",
        },
    }
}


# Define merged categories and labels
MAST_MERGED_CATEGORIES = {
    "Specification Issues": ["1.1+1.2", "1.3", "1.5"],
    "Communication Misalignment": ["1.4+2.1+2.5", "2.2", "2.3", "2.4", "2.6"],
    "Task Verification": ["3.1", "3.2", "3.3"],
}

MAST_MERGED_LABELS = {
    "1.1+1.2": "Disobey specification",
    "1.3": "Step repetition",
    "1.5": "Unaware of termination conditions",
    "1.4+2.1+2.5": "Context loss",
    "2.2": "Fail to ask for clarification",
    "2.3": "Task derailment",
    "2.4": "Information withholding",
    "2.6": "Reasoning-action mismatch",
    "3.1": "Premature termination",
    "3.2": "No or incomplete verification",
    "3.3": "Incorrect verification",
}

# Mapping from original modes to merged modes
MAST_MODE_MAPPING = {
    "1.1": "1.1+1.2",
    "1.2": "1.1+1.2",
    "1.4": "1.4+2.1+2.5",
    "2.1": "1.4+2.1+2.5",
    "2.5": "1.4+2.1+2.5",
    # Keep others as-is
    "1.3": "1.3",
    "1.5": "1.5",
    "2.2": "2.2",
    "2.3": "2.3",
    "2.4": "2.4",
    "2.6": "2.6",
    "3.1": "3.1",
    "3.2": "3.2",
    "3.3": "3.3"
}


def make_failure_prompt(trace: str, task_description: Optional[str] = None, **kwargs):
    # Convert FAILURE_MODE_RUBRIC to a string format
    rubric_text = ""
    for mode, details in FAILURE_MODE_RUBRIC.items():
        rubric_text += f"\n{mode}:\n"
        rubric_text += f"  Definition: {details['definition']}\n"
        rubric_text += f"  Indicators: {', '.join(details['indicators'])}\n"
        rubric_text += f"  Examples: {', '.join(details['examples'][:2])}\n"
    
    prompt = f"""You are an expert at analyzing failed terminal/bash execution traces from AI agents attempting to complete tasks.

        FAILURE MODE RUBRIC:
        {rubric_text}

        EXECUTION TRACE:
        {trace}

        TASK DESCRIPTION:
        {task_description} if provided, otherwise "Not provided"

        Analyze this execution trace and classify the final failure that lead to an error into every category from the rubric above.

        For each rubric category, output a JSON object that follows OUTPUT_MODE_RUBRIC exactly:
        - "score": Float between 0.0 and 1.0 (use 0.0 if the category does not apply).
        - "evidence": String. Direct quote or description from the trace that supports this classification (empty string if score = 0.0). It should be the final failure that lead to an error.
        - "required_skill": String. The skill the agent would need to avoid this failure (fill in even if score = 0.0, using the rubric definition).

        Return your answer strictly in this JSON structure:
        {OUTPUT_MODE_RUBRIC}
        
        Only return valid JSON, no explanations outside of it.
        """
    return prompt


TIMEOUT_FAILURE_RUBRIC = {
    "Timeout: Long-running computation": {
        "definition": "Process exceeded time budget",
        "indicators": ["compiling kernel", "running test suite"],
        "examples": ["compiling kernel", "running test suite"],
        "required_skill": "Estimate runtime / chunk long tasks"
    },
    "Timeout: Stuck waiting for I/O": {
        "definition": "Agent blocked on external input or network",
        "indicators": ["waiting for login", "wget hangs"],
        "examples": ["waiting for login", "wget hangs"],
        "required_skill": "Detect stalled I/O, abort & retry"
    },
    "Timeout: Infinite retry loop": {
        "definition": "Agent retries same command until timeout",
        "indicators": ["retrying same command repeatedly"],
        "examples": ["retrying same command repeatedly"],
        "required_skill": "Loop detection / adaptive retry"
    }
}

def make_timeout_prompt(trace: str, task_description: Optional[str] = None):
    """Create a timeout-specific failure analysis prompt."""
    rubric_text = ""
    for mode, details in TIMEOUT_FAILURE_RUBRIC.items():
        rubric_text += f"\n{mode}:\n"
        rubric_text += f"  Definition: {details['definition']}\n"
        rubric_text += f"  Indicators: {', '.join(details['indicators'])}\n"
        rubric_text += f"  Examples: {', '.join(details['examples'])}\n"
        rubric_text += f"  Required Skill: {details['required_skill']}\n"
    
    prompt = f"""You are an expert at analyzing timeout failures in terminal/bash execution traces from AI agents.

TIMEOUT FAILURE MODE RUBRIC:
{rubric_text}

EXECUTION TRACE:
{trace}

TASK DESCRIPTION:
{task_description if task_description else "Not provided"}

Analyze this timeout failure trace and classify it into the timeout-specific categories above.

For each timeout failure category, output a JSON object with:
- "score": Float between 0.0 and 1.0 (use 0.0 if the category does not apply).
- "evidence": String. Direct quote or description from the trace that supports this classification (empty string if score = 0.0).
- "required_skill": String. The skill the agent would need to avoid this failure.

Return your answer strictly in this JSON structure:
{{
  "failure_modes": {{
    {', '.join([f'"{mode}": {{"score": 0.0, "evidence": "", "required_skill": "{details["required_skill"]}"}}' for mode, details in TIMEOUT_FAILURE_RUBRIC.items()])}
  }}
}}

Only return valid JSON, no explanations outside of it.
"""
    return prompt


def make_mast_prompt(
    trace: str,
    task_description: str,
    definitions: Optional[str] = None,
    examples: Optional[str] = None,
):
    if definitions is None:
        definitions = open(BASE_DIR / "taxonomies/mast/definitions.txt", "r").read()
    if examples is None:
        examples = open(BASE_DIR / "taxonomies/mast/examples.txt", "r").read()

    prompt = (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.5 Ignored Other Agent's Input: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 No or Incorrect Verification: <yes or no>"
        "3.3 Weak Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"
        "Here is the trace: \n"
        f"{trace}"
        "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{definitions} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        f"{examples}"
    )
    return prompt


def parse_failure_definitions(text: str, indent: int = 2):
    """
    Parse a taxonomy definitions.txt file and build a structured JSON schema template.

    Each failure mode (e.g., "1.1 Disobey Specification (Process Compliance)") becomes:
        {
          "label": "<yes | no | unclear>",
          "evidence": "<justification>",
          "confidence_score": "<float>"
        }
    """

    # with open(definitions_path, "r") as f:
    #    text = f.read()

    # Match patterns like "1.1 Disobey Specification (Process Compliance)"
    pattern = r"(?m)^\s*(\d+\.\d+)\s+([A-Za-z–\-\s]+)\s*(?:\([^)]+\))?"
    matches = re.findall(pattern, text)

    failure_modes = {}
    for number, title in matches:
        key = f"{number.strip()} {title.strip()}"
        failure_modes[key] = {
            "label": "<yes | no | unclear>",
            "evidence": "<justification>",
            "confidence_score": "<float>",
        }

    # Assemble final template
    schema = {
        "summary": "<1–5 sentence factual summary of observed problems or inefficiencies, "
        "optionally citing short quotes or episode references "
        "(e.g., 'Episode 3: command not found', 'Episode 5: declared success before running tests').>",
        "task_completed": "<yes | no | unclear>",
        "failure_modes": failure_modes,
    }

    # print(json.dumps(schema, indent=indent, ensure_ascii=False))
    return schema


def parse_failure_definitions2(text: str, indent: int = 2) -> Dict[str, Any]:
    """
    Parse a taxonomy definitions.txt file with multi-line sections.
    Produces structured JSON entries for each failure mode.
    """

    # Match each failure mode header (e.g., "1.1 Disobey Specification (Process Compliance)")
    pattern = r"(?m)^(?P<code>\d+\.\d+)\s+(?P<name>[A-Za-z0-9’'–—\-\s]+?)\s*\((?P<category>[^)]+)\)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError("No failure mode definitions found. Check formatting.")

    failure_modes = {}

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()

        def extract(label):
            """Extract multiline block following 'Label:'"""
            pat = rf"{label}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|\Z)"
            found = re.search(pat, section, re.S)
            if not found:
                return "" if label in ["Decision Rule", "Definition"] else []
            text_block = found.group(1).strip()
            if label in [
                "Anchor Evidence",
                "NOT a failure if",
                "Counterexample (Success)",
            ]:
                # Split bullet points
                bullets = [
                    re.sub(r"^\s*-\s*", "", l.strip())
                    for l in text_block.splitlines()
                    if l.strip()
                ]
                return bullets
            return " ".join(line.strip() for line in text_block.splitlines())

        failure_modes[m.group("code") + " " + m.group("name")] = {
            "label": "<yes | no | unclear>",
            "evidence": "<evidence>",
            "confidence_score": "<float>",
        }

    schema = {
        "summary": (
            "<Episode-by-episode factual summary of reasoning, plans, "
            "and commands for debugging.>"
        ),
        "task_completed": "<yes | no | unclear>",
        "failure_modes": failure_modes,
    }

    # Optional: pretty-print for sanity check
    # print(json.dumps(schema, indent=indent, ensure_ascii=False))
    return schema


def make_tb0_prompt(
    trace: str,
    task_description: str,
    definitions: Optional[str] = None,
    examples: Optional[str] = None,
    reward: Optional[float] = None,
    verifier_data: Optional[Dict[str, Any]] = None,
):
    if definitions is None:
        definitions = open(BASE_DIR / "taxonomies/tb_v0/definitions.txt", "r").read()
    if examples is None:
        examples = open(BASE_DIR / "taxonomies/tb_v0/examples.txt", "r").read()

    # Format verifier data if provided
    if verifier_data is not None:
        summaries = []
        for fname, content in verifier_data.items():
            # Skip reward or invalid entries
            if not isinstance(content, str) or "reward" in fname.lower():
                continue

            content = content.strip()
            if not content:
                continue

            summaries.append(f"**{fname}:**\n```\n{content}\n```")

        verifier_section = "\n\n### Verifier Information\n" + (
            "\n\n".join(summaries) if summaries else "No verifier output provided."
        )
    else:
        verifier_section = ""

    # Format reward information
    reward_section = ""
    if reward is not None:
        if reward == 1.0:
            reward_section = "\n\n### Task Reward\nTask completed successfully according to verifier."
        elif reward == 0.0:
            reward_section = "\n\n### Task Reward\nTask failed according to verifier."
        else:
            raise ValueError(f"Invalid reward: {reward}")
            # reward_section = "\n\n### Task Reward\nTask partially completed according to verifier."

    schema = parse_failure_definitions(definitions)

    prompt = f"""
You are an expert in analyzing and evaluating command-line (CLI) traces of autonomous agents performing tasks in an environment.
Your goal is to identify and classify **failure modes** based solely on observable behavior in the trace.
Ground all judgments in **explicit textual or behavioral evidence** visible in the trace.

---

### Trace Composition

A trace is a sequence of **episodes** representing reasoning–action–feedback cycles.

Episode 0 establishes the starting context for the entire trace (task description and initial environment).
Each subsequent episode (1, 2, …) records one cycle: terminal output, agent analysis/plan, commands for next episode.

Each episode contains:

1. **Terminal Output**
   The raw CLI output produced by the environment in response to the agent’s previous commands.

2. **Agent Response** JSON with:
   * "analysis" : the agent’s interpretation of the terminal state and progress.
   * "plan": rationale for the following actions.
   * "commands" : a list of command objects (keystrokes and timing).
   * "task_complete" *(optional)* : a flag indicating whether the task finished according to the verifier.

Together, **Episode 0** and the sequence of subsequent episodes form a complete **interaction trace**, illustrating how the agent observes, reasons, and acts over time to accomplish the defined terminal task.

---

You must analyze both:
- **Episode-level behavior** (within-step performance)  
- **Cross-episode patterns** (consistency and evolution over time)

---
### Task Outcome and Verifier Context
{reward_section}{verifier_section}

- Task outcome and verifier output are **reference only** to confirm task success or failure.  
- Do **not** extract reasoning errors or commands from verifier logs.  
- Base all judgments strictly on the **agent’s trace (Episodes 0–N)**.
---

###  Evaluation Objective

1. Examine the provided trace carefully.
2. Identify any *observable* signs of failure modes or inefficiencies as defined below.
3. Assess overall task completion** using both the trace and verifier signal.
4. Summarize your findings (1-20 sentences).
5. For **each** failure mode, record:
    - "label": (`yes`, `no`, or `unclear`)
    - "evidence":  a short justification for your label — cite concrete trace content if `"yes"`, 
      or briefly explain why evidence is absent or insufficient if `"no"` or `"unclear"`.
    - "confidence_score": a float in `[0.0, 1.0]` reflecting certainty.


> Mark `"yes"` only when clear textual evidence exists.  
> Do **not** infer hidden reasoning, unstated intentions, or unseen system events.

---

### Evaluation & Scoring Checklist

Follow this checklist *in order* for every trace you evaluate:

A. **Confirm Task Context**

   * [ ] Read all episodes carefully (prompt, output, agent response).
   * [ ] Note whether the *verifier information* is provided and record its value.
   * [ ] Identify the final agent declaration (e.g., “task complete,” “done,” “success”).

B. **Determine Task Completion**

   Use the information from the **Task Reward** section (if present) and the trace itself.
   * [ ] If the reward text says the task *succeeded*, mark `"task_completed": "yes"`.
   * [ ] If it says the task *failed*, mark `"task_completed": "no"`.
   * [ ] If no reward text is present or outcome unclear from the trace, mark `"task_completed": "unclear"`.
   * [ ] Add a short factual explanation in `"summary"` (e.g., “no reward info,” “trace truncated,” “verifier output missing”).

C. **Evidence-Based Reasoning**

   * [ ] Mark "yes" *only* when the trace contains direct evidence (command, log, or textual observation).
   * [ ] Never infer hidden reasoning, internal states, or unobserved events.
   * [ ] Use "unclear" if evidence is partial, indirect, or ambiguous.
   * [ ] Avoid overconfident "yes" when evidence is weak.

D. **Confidence Scoring**

   * [ ] Assign a float between 0.0–1.0 reflecting certainty:

     * 1.0: strong, explicit, repeated evidence
     * 0.7: moderate, clear evidence
     * 0.5: partial or single-instance evidence
     * <0.5: speculative or weak signal
   * [ ] Confidence must align with how clear the evidence is in the trace.

E. **Failure Mode Classification**

   * [ ] Evaluate every failure mode independently. Multiple "yes" labels are allowed.
   * [ ] Refer to the definitions for criteria and anchor evidence.
   * [ ] Use "yes", "no", or "unclear" for each category.
   * [ ] If multiple apply, mark all relevant ones; do not force exclusivity.

F. **System-Level Exclusions**

   * [ ] Do not penalize the agent for:
     * Sandbox or runtime timeouts
     * Environment resets or external interruptions
     * Missing files or states caused by the system, not the agent.

G. **Output Format Requirements**

   * [ ] Return only valid JSON in the specified schema.
   * [ ] All keys and labels must be lowercase ("yes", "no", "unclear").
   * [ ] Each "evidence" field should cite one concrete observation from the trace (e.g., "file not found", "declared success before running tests").
   * [ ] Keep the "summary" concise (1–5 factual sentences).
   * [ ] If all labels = "no", the summary must state "no failures or inefficiencies detected."

H. **Cross-Check Consistency**

   * [ ] Ensure "summary" agrees with "task_completed".
   * [ ] Confidence values align with evidence strength.
   * [ ] No contradictions between failure mode labels or between summary and evidence.

---

### Output Format

**CRITICAL REQUIREMENTS:**
1. Return ONLY valid JSON - no explanatory text before or after
2. Use double quotes for ALL strings (not single quotes like Python dicts)
3. Ensure all JSON keys and string values are properly quoted
4. The response must start with {{ and end with }}
5. Do not wrap the JSON in markdown code blocks

Return only valid JSON using the structure below:
{schema}

---

### Definitions of Failure Modes
{definitions}
---

### Terminal Traces

{trace}

---

Now output the JSON response described above — and **nothing else**. Remember: valid JSON with double quotes only, starting with {{ and ending with }}.

"""
    return prompt


def make_tb1_prompt(
    trace: str,
    task_description: str,
    definitions: Optional[str] = None,
    examples: Optional[str] = None,
    reward: Optional[float] = None,
    verifier_data: Optional[Dict[str, Any]] = None,
):
    """Create TB1 prompt with cleaner separation between system and user messages.

    This prompt assumes more context is in the system message, so the user prompt
    is more concise and focused on the specific task and trace.
    """

    if definitions is None:
        definitions = open(BASE_DIR / "taxonomies/tb_v0/definitions.txt", "r").read()
    if examples is None:
        examples = open(BASE_DIR / "taxonomies/tb_v0/examples.txt", "r").read()

    # Parse definitions to get the schema
    schema = parse_failure_definitions(definitions)

    # Format verifier data concisely
    verifier_info = []
    if verifier_data:
        for fname, content in verifier_data.items():
            if (
                isinstance(content, str)
                and "reward" not in fname.lower()
                and content.strip()
            ):
                verifier_info.append(f"{fname}: {content[:200].strip()}")

    # Format reward
    if reward == 1.0:
        reward_text = "Task completed successfully according to verifier."
    elif reward == 0.0:
        reward_text = "Task failed according to verifier."
    else:
        raise ValueError(f"Invalid reward: {reward}")

    prompt = f"""{reward_text}
{("Verifier: " + " | ".join(verifier_info)) if verifier_info else ""}

## Failure Mode Definitions
{definitions}

## Examples of Failure Modes
{examples}

## Expected JSON Output
{json.dumps(schema, indent=2)}

## Trace
{trace}

Analyze the trace for failure modes. Output only valid JSON matching the schema above."""

    return prompt


def make_tb1_refined_prompt(
    trace: str,
    task_description: str,
    definitions: Optional[str] = None,
    examples: Optional[str] = None,
    reward: Optional[float] = None,
    verifier_data: Optional[Dict[str, Any]] = None,
):
    """Create TB1 refined prompt with targeted examples for problematic modes.

    Uses refined examples focusing on quality over quantity.
    """

    if definitions is None:
        definitions = open(BASE_DIR / "taxonomies/tb_v0/definitions.txt", "r").read()
    if examples is None:
        # Use refined examples instead of comprehensive ones
        examples = open(BASE_DIR / "taxonomies/tb_v0/examples_refined.txt", "r").read()

    # Parse definitions to get the schema
    schema = parse_failure_definitions(definitions)

    # Format verifier data concisely
    verifier_info = []
    if verifier_data:
        for fname, content in verifier_data.items():
            if (
                isinstance(content, str)
                and "reward" not in fname.lower()
                and content.strip()
            ):
                verifier_info.append(f"{fname}: {content[:200].strip()}")

    # Format reward
    if reward == 1.0:
        reward_text = "Task completed successfully according to verifier."
    elif reward == 0.0:
        reward_text = "Task failed according to verifier."
    else:
        raise ValueError(f"Invalid reward: {reward}")

    prompt = f"""{reward_text}
{("Verifier: " + " | ".join(verifier_info)) if verifier_info else ""}

## Failure Mode Definitions
{definitions}

## Examples
{examples}

## Expected JSON Output
{json.dumps(schema, indent=2)}

## Trace
{trace}

## End of Trace

Analyze the trace for failure modes. Output only valid JSON matching the schema above."""

    return prompt


def make_tb2_prompt(
    trace: str,
    task_description: Optional[str] = None,
    definitions: Optional[str] = None,
    examples: Optional[str] = None,
    reward: Optional[float] = None,
    verifier_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Create TB2 prompt that relies on system message for defs/examples/schema."""

    # Allow callers to pass file paths for any text inputs
    def _load_if_file(value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return value
        try:
            from pathlib import Path

            p = Path(value)
            if p.exists() and p.is_file():
                return p.read_text()
        except Exception:
            pass
        return value

    task_description = _load_if_file(task_description)
    definitions = _load_if_file(definitions)
    examples = _load_if_file(examples)

    # Build header information
    header_parts = []

    # Add reward information
    if reward is not None:
        if reward == 1.0:
            header_parts.append("Task completed successfully according to verifier.")
        elif reward == 0.0:
            header_parts.append("Task failed according to verifier.")

    # Add verifier data
    if verifier_data:
        verifier_info = []
        for fname, content in verifier_data.items():
            if (
                not isinstance(content, str)
                or not content.strip()
                or "reward" in fname.lower()
            ):
                continue

            # Extract short test summary info, skip if not present
            short_test_summary_info = "=========================== short test summary info ============================"
            if short_test_summary_info in content:
                summary_lines = content.split(short_test_summary_info)[1].split("\n")[
                    1:
                ]
                verifier_info.append("\n".join(summary_lines))

        if verifier_info:
            header_parts.append("Verifier Information: " + "\n".join(verifier_info))

    # Build the complete prompt
    prompt_parts = []

    # Add header
    if header_parts:
        prompt_parts.extend(header_parts)
        prompt_parts.append("")  # Empty line after header

    # Add task description if provided
    if task_description:
        prompt_parts.append(f"## Task Description\n{task_description}\n")

    # Add definitions if provided
    if definitions:
        prompt_parts.append(f"## Failure Mode Definitions\n{definitions}\n")

    # Add examples if provided
    if examples:
        prompt_parts.append(f"## Examples of Failure Modes\n{examples}\n")

    # Add trace
    prompt_parts.append(f"## Trace\n{trace}\n")
    prompt_parts.append("## End of Trace\n")

    # Add instruction
    prompt_parts.append(
        "## Analyze the trace for failure modes. Output only valid JSON matching the schema above."
    )

    return "\n".join(prompt_parts)


MAKE_FAILURE_PROMPTS = {
    "base": make_failure_prompt,
    "mast": make_mast_prompt,
    "timeout": make_timeout_prompt,
    "tb0": make_tb0_prompt,
    "tb1": make_tb1_prompt,
    "tb1_refined": make_tb1_refined_prompt,
    "tb2": make_tb2_prompt,
}


def parse_response_base(response: str) -> Dict[str, any]:
    """Parse base response with improved error handling and JSON extraction."""
    # Strip markdown code blocks if present
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    # Try to extract dictionary if there's extra text
    json_obj = extract_dict_from_text(cleaned_response)
    if json_obj is None:
        # If extraction failed, try direct parsing
        try:
            result = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            breakpoint()
            print(f"ERROR: Failed to parse JSON: {e}")
            print(f"Cleaned response was: {repr(cleaned_response[:500])}...")
            # Return empty failure modes as fallback
            print("WARNING: Returning empty failure modes due to parse error")
            return {}
    else:
        result = json_obj
    
    failure_modes = result.get("failure_modes", {})
    # Validate keys if we got valid failure modes
    if failure_modes and hasattr(OUTPUT_MODE_RUBRIC.get("failure_modes", {}), "keys"):
        invalid_keys = [
            key
            for key in failure_modes.keys()
            if key not in OUTPUT_MODE_RUBRIC["failure_modes"].keys()
        ]
        if invalid_keys:
            print(f"WARNING: Invalid failure mode keys found: {invalid_keys}")
    return failure_modes

def parse_response_mast(response: str) -> Tuple[Dict[str, any], str]:
    """Parse MAST response to extract scores and full analysis text.
    
    Returns:
        Tuple of (failure_modes_dict, full_analysis_text)
        where failure_modes_dict contains scores for each mode
        and full_analysis_text contains the complete A, B, C sections
    """
    try:
        # Initialize failure modes dict
        failure_modes = {}
        mode_list = ['1.1', '1.2', '1.3', '1.4', '1.5',
                     '2.1', '2.2', '2.3', '2.4', '2.5', '2.6',
                     '3.1', '3.2', '3.3']
        
        # Clean up the response - remove @@ markers if present
        cleaned_response = response.strip()
        if cleaned_response.startswith('@@'):
            cleaned_response = cleaned_response[2:]
        if cleaned_response.endswith('@@'):
            cleaned_response = cleaned_response[:-2]
        
        # Store the full cleaned response as the analysis text
        full_analysis = cleaned_response.strip()
        
        # Process each failure mode to extract scores
        for mode in mode_list:
            # Various patterns to match different response formats
            patterns = [
                # Format with C. prefix and colon
                rf"C\..*?{mode}.*?(yes|no)",
                # Format with just C prefix without dot
                rf"C{mode}\s+(yes|no)",
                # Format with mode directly (with or without spaces)
                rf"{mode}\s*[:]\s*(yes|no)",
                rf"{mode}\s+(yes|no)",
                # Format with newlines
                rf"{mode}\s*\n\s*(yes|no)",
                # Format with C prefix and newlines
                rf"C\.{mode}\s*\n\s*(yes|no)"
            ]

            found = False
            score = 0
            
            for pattern in patterns:
                matches = re.findall(
                    pattern, cleaned_response, re.IGNORECASE | re.DOTALL
                )
                if matches:
                    # Use the first match
                    score = 1 if matches[0].lower() == "yes" else 0
                    found = True
                    break

            if not found:
                # If we still can't find a match, try a more general approach
                general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                match = re.search(
                    general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL
                )

                if match:
                    score = 1 if match.group(1).lower() == "yes" else 0
                    found = True

            if not found:
                # If all attempts fail, default to 'no'
                print(f"Warning: Could not find mode {mode} in response")
                score = 0

            # Store just the score for backward compatibility
            # Evidence and required_skill are now in the full_analysis text
            failure_modes[mode] = {
                "score": float(score),
                "evidence": "",  # Will be extracted from full_analysis when needed
                "required_skill": "",  # Will be extracted from full_analysis when needed
            }

    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")
    return failure_modes, full_analysis


def parse_response_timeout(response: str) -> Dict[str, Any]:
    """Parse timeout response with improved error handling."""
    # Clean up the response - remove markdown if present
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    # Try to extract dictionary if there's extra text
    json_obj = extract_dict_from_text(cleaned_response)
    if json_obj is None:
        try:
            result = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse timeout JSON: {e}")
            print(f"Response was: {repr(cleaned_response[:500])}...")
            # Return default structure on parse error
            return {
                mode: {"score": 0, "evidence": "", "required_skill": ""}
                for mode in TIMEOUT_FAILURE_RUBRIC.keys()
            }
    else:
        result = json_obj

    failure_modes = result.get("failure_modes", {})

    expected = set(TIMEOUT_FAILURE_RUBRIC.keys())
    actual = set(failure_modes.keys())

    missing = expected - actual
    extra = actual - expected

    if missing:
        print(f"WARNING: Missing failure mode(s): {missing}, adding with score=0")
        for mode in missing:
            failure_modes[mode] = {"score": 0, "evidence": "", "required_skill": ""}
    if extra:
        print(f"Warning: Unexpected failure mode(s): {extra}")

    return failure_modes


def parse_response_tb0(response: str) -> Tuple[Dict[str, any], str]:
    """Parse TB0 response to extract failure modes and analysis.

    This function uses ast.literal_eval to handle Python dict format responses
    as well as JSON format responses.

    Returns:
        Tuple of (failure_modes_dict, full_analysis_text)
    """
    try:
        # Extract dictionary from response (handles both JSON and Python dict format)
        result = extract_dict_from_text(response)

        if result is None:
            # If extraction completely failed, try to extract key info with regex
            print("Warning: Could not parse TB0 response, attempting regex extraction")
            print(f"Response preview: {response[:200]}...")

            # Try to extract at least the summary if present
            summary_match = re.search(
                r"['\"]summary['\"]\s*:\s*['\"]([^'\"]*)['\"]", response
            )
            task_match = re.search(
                r"['\"]task_completed['\"]\s*:\s*['\"]([^'\"]*)['\"]", response
            )

            summary = (
                summary_match.group(1) if summary_match else "Failed to parse response"
            )
            task_completed = task_match.group(1) if task_match else "unclear"

            # Return empty failure modes with the extracted info
            return (
                {},
                f"Summary: {summary}\nTask Completed: {task_completed}\n[Parse Error: Could not extract full response]",
            )

        # Extract summary and task_completed
        summary = result.get("summary", "")
        per_episode_summary = result.get("per_episode_summary", "")
        task_completed = result.get("task_completed", "unclear")

        # Extract failure modes
        failure_modes_raw = result.get("failure_modes", {})

        # Convert to standardized format
        failure_modes = {}
        for mode_name, mode_data in failure_modes_raw.items():
            # Handle case where mode_data might be a string or other simple type
            if not isinstance(mode_data, dict):
                mode_data = {"label": str(mode_data)}

            # Extract mode number (e.g., "1.1" from "1.1 Disobey Specification")
            mode_num = mode_name.split()[0] if " " in mode_name else mode_name

            # Convert label to score
            label = str(mode_data.get("label", "no")).lower()
            if label in ["yes", "true", "1"]:
                score = 1.0
            elif label in ["unclear", "maybe", "partial"]:
                score = 0.5
            else:
                score = 0.0

            # Use confidence_score if provided, otherwise use converted score
            confidence = mode_data.get("confidence_score", score)

            failure_modes[mode_num] = {
                "score": score,
                "confidence": confidence,
                "evidence": mode_data.get("evidence", "")
                if isinstance(mode_data, dict)
                else "",
            }

            if mode_data.get("anchors"):
                failure_modes[mode_num]["anchors"] = mode_data.get("anchors", [])
            if mode_data.get("rationale"):
                failure_modes[mode_num]["rationale"] = mode_data.get("rationale", "")
            if mode_data.get("required_skill"):
                failure_modes[mode_num]["required_skill"] = mode_data.get(
                    "required_skill", ""
                )

        # Create full analysis text combining all information
        full_analysis = f"Overall Summary: {summary}\n Per-Episode Summary: {per_episode_summary}\n Task Completed: {task_completed}\n"

        return failure_modes, full_analysis

    except Exception as e:
        print(f"ERROR: Unexpected error parsing TB0 response: {e}")
        print(f"Response was: {repr(response[:300])}")
        # Return empty failure modes with error information
        return {}, f"[Parse Error: {str(e)}]\n"


PARSE_RESPONSE_FUNCTIONS = {
    "base": parse_response_base,
    "mast": parse_response_mast,
    "tb0": parse_response_tb0,
    "tb1": parse_response_tb0,
    "tb1_refined": parse_response_tb0,  # Uses same parser as tb0/tb1
    "tb1_mode22": parse_response_tb0,  # Uses same parser for Mode 2.2 enhanced
    "tb1_mode22_correct": parse_response_tb0,  # Uses same parser for corrected Mode 2.2
    "tb2": parse_response_tb0,
    "timeout": parse_response_timeout,
}

if __name__ == "__main__":
    trace = "Task: copy file A to B \n "

    prompt = MAKE_FAILURE_PROMPTS["mast"](trace)
    print(prompt)
