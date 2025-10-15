#!/usr/bin/env python3
"""
Analyze failed trials using failure classification.

This script processes trial traces using various failure classifiers (judge or embedder)
and stores results in a database. Supports parallel processing with multiple judges.
"""

import argparse
import json
import os
import re
import sys
import threading
import yaml
import weave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from failure_classifier import FAILURE_CLASSIFIERS
from judge_database import JudgeResultsDB
from trace_parser import TraceParser


@dataclass
class ClassifierConfig:
    """Configuration for the failure classifier."""
    failure_prompt_type: str
    judges: List[str]


@dataclass
class AnalysisArgs:
    """Analysis arguments extracted from config."""
    task_ids: List[str]
    failure_classifier: str
    classifier_config: ClassifierConfig
    database_output: bool
    database_path: Optional[str]
    max_workers: int
    weave_enabled: bool
    weave_project: Optional[str]


def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable expansion."""
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Replace ${oc.env:VAR_NAME} with environment variables
    pattern = r'\$\{oc\.env:([^}]+)\}'
    def replace_env(match):
        env_var = match.group(1)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ValueError(f"Environment variable {env_var} not found")
        return env_value
    
    content = re.sub(pattern, replace_env, content)
    return yaml.safe_load(content)


def get_trial_trace(trace_parser: TraceParser, trial_id: str) -> Tuple[str, Optional[float], Optional[Dict[str, Any]]]:
    """Get trace text and metadata for a trial ID.
    
    Returns:
        Tuple of (trace_text, reward, verifier_data)
    """
    try:
        trace = trace_parser.parse_trace(trial_id)
        trace_text = trace.to_json(include_metadata=False)
        reward = trace.get_reward()
        verifier_data = trace.verifier_data
        return trace_text, reward, verifier_data
    except Exception as e:
        print(f"Warning: Could not parse trace for trial {trial_id}: {e}")
        return "", None, None


def save_trial_to_database(
    trial_id: str,
    trial_data: Dict[str, Any],
    judge_name: str,
    database_config: Dict[str, Any]
):
    """Save a single trial's results to the database immediately."""
    db_path = database_config['path']
    failure_classifier = database_config['failure_classifier']
    failure_prompt_type = database_config['failure_prompt_type']
    
    db = JudgeResultsDB(db_path)
    
    try:
        # Extract metadata from trace files
        trace_dir = BASE_DIR / "traces" / trial_id
        
        if not trace_dir.exists():
            raise FileNotFoundError(f"Trial trace directory not found: {trace_dir}")
        
        # Load metadata from config.json
        config_path = trace_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Trial config.json not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract agent.name and agent.model_name from config
        agent_config = config.get('agent', {})
        model_name = agent_config.get('model_name', 'unknown')
        agent_name = agent_config.get('name', 'unknown')
        
        # Clean up model name if it has a prefix like "openai/"
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        
        # Load task name from result.json if available
        task_name = None
        result_path = trace_dir / "result.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                result = json.load(f)
                task_name = result.get('task_name')
        
        # First insert the trial if it doesn't exist
        db.insert_trial(
            trial_id=trial_id,
            task_name=task_name,
            agent_name=agent_name,
            model_name=model_name,
            trace_length=trial_data.get('trace_length')
        )
        
        if 'failure_modes' in trial_data:
            failure_modes_dict = format_failure_modes(trial_data['failure_modes'])
            
            # Store successful result with full analysis if available
            db.store_judgment_result(
                trial_id=trial_id,
                failure_classifier=failure_classifier,
                judge_model=judge_name,
                failure_prompt_type=failure_prompt_type,
                failure_modes=failure_modes_dict,
                failure_mode_analysis=trial_data.get('full_analysis', ''),
                trace_length=trial_data.get('trace_length')
            )
        elif 'error' in trial_data:
            # Store error result
            db.store_judgment_result(
                trial_id=trial_id,
                failure_classifier=failure_classifier,
                judge_model=judge_name,
                failure_prompt_type=failure_prompt_type,
                error_message=trial_data['error'],
                trace_length=trial_data.get('trace_length')
            )
    finally:
        db.close()


def process_single_trial(
    classifier, 
    trial_id: str, 
    results_lock: threading.Lock, 
    results: Dict[str, Any],
    failure_prompt_type: str = 'mast',
    weave_enabled: bool = False,
    database_config: Optional[Dict[str, Any]] = None,
    judge_name: str = None
) -> Dict[str, Any]:
    """Process a single trial with the classifier and save to database immediately."""
    try:
        # Check if already processed (thread-safe)
        with results_lock:
            if trial_id in results["trials"]:
                return {"trial_id": trial_id, "status": "skipped", "reason": "already_processed"}
        
        # Get trace text and metadata
        trace_parser = TraceParser(BASE_DIR / "traces")
        trace_text, reward, verifier_data = get_trial_trace(trace_parser, trial_id)
        if not trace_text:
            print(f"Warning: Empty trace for trial {trial_id}", flush=True)
            return {"trial_id": trial_id, "status": "error", "reason": "empty_trace"}
        
        # Classify the failure with additional error handling
        error_msg = None
        try:
            result = classifier.process_trace(
                trace=trace_text, 
                trial_id=trial_id,
                reward=reward,
                verifier_data=verifier_data
            )
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for trial {trial_id}: {e}",flush=True)
            # Mark as error, not empty success
            result = None
            error_msg = f"JSON parsing failed: {str(e)}"
        except Exception as e:
            print(f"Classification error for trial {trial_id}: {e}", flush=True)
            # Mark as error, not empty success
            result = None
            error_msg = f"Classification failed: {str(e)}"
        
        # Handle errors vs successful results
        if error_msg:
            # Store error result
            trial_data = {
                "error": error_msg,
                "trace_length": len(trace_text)
            }
        else:
            # Handle different return formats based on prompt type
            if failure_prompt_type in ['mast', 'tb0'] and isinstance(result, tuple):
                failure_modes, full_analysis = result
            else:
                failure_modes = result if isinstance(result, dict) else {}
                full_analysis = None
            
            # Only store as success if we have actual failure modes or it's explicitly empty
            if failure_modes is not None:
                # Store successful result
                trial_data = {
                    "failure_modes": failure_modes,
                    "trace_length": len(trace_text)
                }
                if full_analysis:
                    trial_data["full_analysis"] = full_analysis
            else:
                # Treat None failure_modes as an error
                trial_data = {
                    "error": "No failure modes returned from classifier",
                    "trace_length": len(trace_text)
                }
            
        with results_lock:
            results["trials"][trial_id] = trial_data
        
        # Save to database immediately if configured
        if database_config and database_config.get('enabled'):
            save_trial_to_database(
                trial_id=trial_id,
                trial_data=trial_data,
                judge_name=judge_name,
                database_config=database_config
            )
        
        return {"trial_id": trial_id, "status": "success", "trace_length": len(trace_text)}
        
    except Exception as e:
        # Save error to database if configured
        if database_config and database_config.get('enabled'):
            trial_data = {"error": str(e)}
            save_trial_to_database(
                trial_id=trial_id,
                trial_data=trial_data,
                judge_name=judge_name,
                database_config=database_config
            )
        return {"trial_id": trial_id, "status": "error", "reason": str(e)}


def analyze_task_ids(
    task_ids: List[str], 
    failure_classifier: str,
    failure_prompt_type: str,
    judge_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 5,
    weave_enabled: bool = False,
    database_config: Optional[Dict[str, Any]] = None,
    judge_name: str = None
) -> Dict[str, Any]:
    """
    Analyze task IDs using the specified classifier.
    
    Args:
        task_ids: List of task IDs to analyze
        failure_classifier: Type of classifier to use ("judge" or "embedder") 
        failure_prompt_type: Type of failure prompt ("base", "mast", or "timeout")
        judge_config: Optional judge configuration
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary containing analysis results
    """
    classifier = FAILURE_CLASSIFIERS[failure_classifier](config=judge_config)
    
    # Initialize results structure
    results = {
        "metadata": {
            "failure_classifier": failure_classifier,
            "failure_prompt_type": failure_prompt_type,
            "trial_filter": "user_specified",
            "total_tasks": len(task_ids)
        },
        "trials": {}
    }
    
    total_trials = len(task_ids)
    print(f"Processing {total_trials} trials with {failure_classifier} classifier "
          f"({failure_prompt_type} prompt) using {max_workers} workers", flush=True)
    
    # Thread-safe processing
    results_lock = threading.Lock()
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_trial = {
            executor.submit(
                process_single_trial, 
                classifier, 
                trial_id, 
                results_lock, 
                results,
                failure_prompt_type,
                weave_enabled,
                database_config,
                judge_name
            ): trial_id
            for trial_id in task_ids
        }
        
        # Process completed tasks
        for future in as_completed(future_to_trial):
            trial_id = future_to_trial[future]
            try:
                result = future.result()
                processed_count += 1
                
                status_msg = f"[{processed_count}/{total_trials}] {trial_id}: "
                if result["status"] == "success":
                    print(f"{status_msg}Success", flush=True)
                elif result["status"] == "skipped":
                    print(f"{status_msg}Skipped ({result['reason']})", flush=True)
                else:
                    print(f"{status_msg}Error - {result['reason']}", flush=True)

                if processed_count % 10 == 0:
                    print(f"Progress: {processed_count}/{total_trials} trials processed", flush=True)
                        
            except Exception as e:
                processed_count += 1
                print(f"[{processed_count}/{total_trials}] {trial_id}: Exception - {e}", flush=True)
    
    success_count = len([t for t in results['trials'].values() if 'failure_modes' in t])
    error_count = len([t for t in results['trials'].values() if 'error' in t])
    
    print(f"\nAnalysis complete!", flush=True)
    print(f"Successfully processed: {success_count} trials", flush=True)
    print(f"Errors encountered: {error_count} trials", flush=True)
    
    return results


def parse_config_to_args(config: Dict[str, Any]) -> AnalysisArgs:
    """Parse configuration dictionary into structured arguments."""
    task_ids = config.get('task_ids', [])
    if not task_ids:
        raise ValueError("task_ids must be specified in the config file")
    
    failure_classifier = config.get('failure_classifier', 'judge')
    
    classifier_config_dict = config.get('classifier_config', {})
    classifier_config = ClassifierConfig(
        failure_prompt_type=classifier_config_dict.get('failure_prompt_type', 'mast'),
        judges=classifier_config_dict.get('judges', [])
    )
    
    # Parse Weave configuration
    weave_config = config.get('weave_config', {})
    weave_enabled = weave_config.get('enabled', False) if weave_config else False
    weave_project = weave_config.get('project_name', None) if weave_config else None
    
    return AnalysisArgs(
        task_ids=task_ids,
        failure_classifier=failure_classifier,
        classifier_config=classifier_config,
        database_output=config.get('database_output', False),
        database_path=config.get('database_path'),
        max_workers=config.get('max_workers', 10),
        weave_enabled=weave_enabled,
        weave_project=weave_project
    )


def check_existing_results(args: AnalysisArgs) -> Dict[str, List[str]]:
    """Check database for existing results and determine what needs processing."""
    db_path = args.database_path
    task_judge_pairs_to_process = {}
    
    if Path(db_path).exists():
        print(f"Using existing database at {db_path}, checking for existing results...", flush=True)
        db = JudgeResultsDB(db_path)
        
        total_tasks_needing_processing = 0
        already_processed_count = 0
        
        for i, task_id in enumerate(args.task_ids, 1):
            judges_needed = []
            for judge_name in args.classifier_config.judges:
                try:
                    has_results = db.has_judge_results(
                        trial_id=task_id,
                        failure_classifier=args.failure_classifier,
                        judge_model=judge_name,
                        failure_prompt_type=args.classifier_config.failure_prompt_type
                    )
                    if not has_results:
                        judges_needed.append(judge_name)
                        total_tasks_needing_processing += 1
                    else:
                        already_processed_count += 1
                except Exception as e:
                    print(f"Error checking task {task_id} with judge {judge_name}: {e}", flush=True)
                    # Assume it needs processing if there's an error
                    judges_needed.append(judge_name)
                    total_tasks_needing_processing += 1
            
            if judges_needed:
                task_judge_pairs_to_process[task_id] = judges_needed
            
            # Log progress every 10 tasks when checking
            if i % 10 == 0 or i == len(args.task_ids):
                print(f"Checked {i}/{len(args.task_ids)} tasks...", flush=True)
        
        db.close()
        
        print(f"\nDatabase check complete:", flush=True)
        print(f"  - Already processed: {already_processed_count} task-judge pairs", flush=True)
        print(f"  - Need processing: {total_tasks_needing_processing} task-judge pairs", flush=True)
        
        if not task_judge_pairs_to_process:
            print("All task-judge pairs already processed! Analysis complete.", flush=True)
            return {}
        
        total_pairs = sum(len(judges) for judges in task_judge_pairs_to_process.values())
        print(f"Will process {total_pairs} task-judge pairs across {len(task_judge_pairs_to_process)} tasks", flush=True)
        
    else:
        print(f"Database does not exist at {db_path}, will create during analysis...", flush=True)
        # If database doesn't exist, process all tasks for all judges
        task_judge_pairs_to_process = {
            task_id: args.classifier_config.judges if args.classifier_config.judges else ["gpt-4o"]
            for task_id in args.task_ids
        }
    
    return task_judge_pairs_to_process


def main(config):
    """Main analysis function."""
    args = parse_config_to_args(config)
    
    # Initialize Weave if enabled
    if args.weave_enabled:
        project_name = args.weave_project or os.getenv("WANDB_PROJECT", "judge")
        print(f"Initializing Weave with project: {project_name}", flush=True)
        weave.init(project_name)
        print("Weave tracking enabled", flush=True)
    else:
        print("Weave tracking disabled", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Starting analysis of {len(args.task_ids)} specified task IDs", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Determine what needs processing
    if args.database_output:
        if not args.database_path:
            raise ValueError("database_path must be specified in config when database_output is true")
        
        task_judge_pairs_to_process = check_existing_results(args)
        if not task_judge_pairs_to_process:
            return
    else:
        # If database output is disabled, process all tasks for all judges
        task_judge_pairs_to_process = {
            task_id: args.classifier_config.judges if args.classifier_config.judges else ["gpt-4o"]
            for task_id in args.task_ids
        }
    
    # Get unique judges and tasks to process
    all_judges = set()
    all_tasks = set()
    for task_id, judges in task_judge_pairs_to_process.items():
        all_tasks.add(task_id)
        all_judges.update(judges)
    
    print(f"Processing {len(all_tasks)} tasks with {len(all_judges)} judges "
          f"using prompt type: {args.classifier_config.failure_prompt_type}", flush=True)
    print(f"Judges: {sorted(all_judges)}", flush=True)
    
    # Process judges in parallel (if multiple judges), then trials within each judge in parallel
    if len(all_judges) > 1:
        max_judge_workers = min(len(all_judges), 3)  # Limit concurrent judges to avoid API rate limits
        print(f"Processing {len(all_judges)} judges in parallel with {max_judge_workers} concurrent judges", flush=True)
        
        with ThreadPoolExecutor(max_workers=max_judge_workers) as executor:
            # Submit all judge processing tasks
            future_to_judge = {
                executor.submit(process_judge, judge_name, task_judge_pairs_to_process, args): judge_name
                for judge_name in sorted(all_judges)
            }
            
            # Process completed judge tasks
            for future in as_completed(future_to_judge):
                judge_name = future_to_judge[future]
                try:
                    future.result()
                    print(f"Judge {judge_name} completed successfully", flush=True)
                except Exception as e:
                    print(f"Judge {judge_name} failed with error: {e}", flush=True)
                    raise e
    else:
        # Single judge - process sequentially but with parallel trials
        judge_name = list(all_judges)[0]
        print(f"Processing single judge {judge_name} with parallel trials", flush=True)
        process_judge(judge_name, task_judge_pairs_to_process, args)
    
    print(f"\n{'='*60}", flush=True)
    print(f"All judges processed successfully", flush=True)
    print(f"{'='*60}", flush=True)


def process_judge(judge_name: str, task_judge_pairs_to_process: Dict[str, List[str]], args: AnalysisArgs):
    """Process tasks for a specific judge."""
    # Get tasks that need processing for this specific judge
    tasks_for_judge = [
        task_id for task_id, judges in task_judge_pairs_to_process.items()
        if judge_name in judges
    ]
    
    if not tasks_for_judge:
        print(f"\n=== Skipping judge {judge_name} - no tasks need processing ===", flush=True)
        return
        
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {len(tasks_for_judge)} tasks with judge: {judge_name}", flush=True)
    print(f"Prompt type: {args.classifier_config.failure_prompt_type}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Create classifier config with the judge name as model name
    classifier_config = {
        "model_name": judge_name,
        "failure_prompt_type": args.classifier_config.failure_prompt_type
    }
    
    # Prepare database configuration for incremental saves
    database_config = None
    if args.database_output:
        database_config = {
            'enabled': True,
            'path': args.database_path,
            'failure_classifier': args.failure_classifier,
            'failure_prompt_type': args.classifier_config.failure_prompt_type
        }
    
    # If Weave is enabled, add a run context for this judge
    if args.weave_enabled:
        run_config = {
            "judge_name": judge_name,
            "prompt_type": args.classifier_config.failure_prompt_type,
            "num_tasks": len(tasks_for_judge)
        }
        with weave.attributes(run_config):
            results = analyze_task_ids(
                tasks_for_judge,
                args.failure_classifier,
                args.classifier_config.failure_prompt_type,
                judge_config=classifier_config,
                max_workers=args.max_workers,
                weave_enabled=args.weave_enabled,
                database_config=database_config,
                judge_name=judge_name
            )
    else:
        results = analyze_task_ids(
            tasks_for_judge,
            args.failure_classifier,
            args.classifier_config.failure_prompt_type,
            judge_config=classifier_config,
            max_workers=args.max_workers,
            weave_enabled=args.weave_enabled,
            database_config=database_config,
            judge_name=judge_name
        )
    print(f"\nAnalysis complete for judge: {judge_name}", flush=True)
    # Note: Results are now saved incrementally during processing


# Removed save_results_to_database function - results are now saved incrementally


def format_failure_modes(raw_failure_modes) -> Dict[str, Dict[str, Any]]:
    """Format failure modes to the expected database format."""
    failure_modes_dict = {}
    
    # Handle the format from parse_response_mast (dict with dicts or lists)
    if isinstance(raw_failure_modes, dict):
        for mode, value in raw_failure_modes.items():
            # Check if value is already a dict with score, evidence, required_skill
            if isinstance(value, dict):
                failure_modes_dict[mode] = {
                    'score': float(value.get('score', 0)),
                    'confidence': float(value.get('confidence', 0)),
                    'evidence': value.get('evidence', ''),
                    'required_skill': value.get('required_skill', '')
                }
            # If value is a list, take the first element as score (backward compatibility)
            elif isinstance(value, list):
                score = value[0] if value else 0
                failure_modes_dict[mode] = {
                    'score': float(score),
                    'confidence': float(score),
                    'evidence': '',
                    'required_skill': ''
                }
            else:
                # If value is a scalar, use it as score
                failure_modes_dict[mode] = {
                    'score': float(value),
                    'confidence': float(value),
                    'evidence': '',
                    'required_skill': ''
                }
    else:
        # If it's already in the right format, use it as is
        failure_modes_dict = raw_failure_modes
    
    return failure_modes_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze failed trials using failure classification"
    )
    parser.add_argument(
        "--config-file", 
        help="Configuration file", 
        default="config/failure_debug_t0_d1.yaml"
    )
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config_file}", flush=True)
    config = load_yaml_config(args.config_file)
    
    main(config)