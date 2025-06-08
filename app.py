import os
import gradio as gr
import requests
import inspect
import pandas as pd
import asyncio
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from agent import AsyncCodeAgentManager 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Async Agent Wrapper ---
class AsyncAgentWrapper:
    """
    Wrapper class that handles async operations for the AsyncCodeAgentManager
    in a synchronous Gradio environment.
    """
    
    def __init__(
        self,
        primary_model_id: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        fallback_model_id: str = 'Qwen/Qwen2.5-Coder-32B-Instruct',
        timeout_seconds: int = 120,
        max_steps: int = 15,
        verbosity_level: int = 2
    ):
        """Initialize the AsyncAgentWrapper with AsyncCodeAgentManager."""
        print("Initializing AsyncAgentWrapper with AsyncCodeAgentManager...")
        
        self.agent_manager = AsyncCodeAgentManager(
            primary_model_id=primary_model_id,
            fallback_model_id=fallback_model_id,
            timeout_seconds=timeout_seconds,
            max_steps=max_steps,
            verbosity_level=verbosity_level
        )
        
        # Create a thread pool for running async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("AsyncAgentWrapper initialized successfully")
    
    def _run_async_in_thread(self, coro):
        """Run an async coroutine in a separate thread with its own event loop."""
        def run_in_new_loop():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        # Submit to thread pool and wait for result
        future = self.executor.submit(run_in_new_loop)
        return future.result()
    
    def __call__(self, question: str) -> str:
        """Process a single question synchronously."""
        print(f"Agent received question: {question}")
        try:
            # Run the async query in a separate thread
            result = self._run_async_in_thread(
                self.agent_manager.run_query(question)
            )
            
            if result['success']:
                response = str(result['response'])
                print(f"Agent response: {response}")
                print(f"Model used: {result['model_used']}")
                print(f"Execution time: {result['total_execution_time']:.2f}s")
                return response
            else:
                error_msg = f"Agent failed: {result['error']}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return error_msg
    
    def run_multiple_questions(self, questions_data: List[Dict]) -> List[Dict]:
        """Process multiple questions efficiently using async capabilities."""
        print(f"Processing {len(questions_data)} questions with async agent...")
        
        try:
            # Extract questions for batch processing
            questions = []
            task_ids = []
            
            for item in questions_data:
                task_id = item.get("task_id")
                question_text = item.get("question")
                if task_id and question_text is not None:
                    questions.append(question_text)
                    task_ids.append(task_id)
            
            if not questions:
                return []
            
            # Run all questions concurrently
            results = self._run_async_in_thread(
                self.agent_manager.run_multiple_queries(questions)
            )
            
            # Format results for submission
            formatted_results = []
            for i, result in enumerate(results):
                task_id = task_ids[i]
                if result['success']:
                    answer = str(result['response'])
                    print(f"Task {task_id}: Success with {result['model_used']} in {result['total_execution_time']:.2f}s")
                else:
                    answer = f"AGENT ERROR: {result['error']}"
                    print(f"Task {task_id}: Failed - {result['error']}")
                
                formatted_results.append({
                    "Task ID": task_id,
                    "Question": questions[i],
                    "Submitted Answer": answer,
                    "Model Used": result.get('model_used', 'Unknown'),
                    "Execution Time": f"{result.get('total_execution_time', 0):.2f}s"
                })
            
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent manager."""
        return self.agent_manager.get_status()
    
    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the AsyncAgentWrapper on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (using AsyncAgentWrapper)
    try:
        agent = AsyncAgentWrapper(
            timeout_seconds=180,  # 3 minutes timeout
            max_steps=15,
            verbosity_level=2
        )
        
        # Print agent status
        status = agent.get_status()
        print(f"Agent Status: {status}")
        
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        logger.error(f"Agent initialization error: {e}")
        return f"Error initializing agent: {e}", None

    # In the case of an app running as a Hugging Face space, this link points toward your codebase
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code URL: {agent_code}")

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent (using batch processing for efficiency)
    print(f"Running async agent on {len(questions_data)} questions...")
    
    try:
        # Use the new batch processing method
        results_log = agent.run_multiple_questions(questions_data)
        
        if not results_log:
            print("Agent did not produce any results.")
            return "Agent did not produce any results.", pd.DataFrame()
        
        # Prepare answers payload for submission
        answers_payload = []
        for result in results_log:
            if not result["Submitted Answer"].startswith("AGENT ERROR:"):
                answers_payload.append({
                    "task_id": result["Task ID"], 
                    "submitted_answer": result["Submitted Answer"]
                })
        
        print(f"Successfully processed {len(results_log)} questions, {len(answers_payload)} valid answers")
        
    except Exception as e:
        error_msg = f"Error during batch processing: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        return error_msg, pd.DataFrame()

    if not answers_payload:
        print("Agent did not produce any valid answers to submit.")
        results_df = pd.DataFrame(results_log)
        return "Agent did not produce any valid answers to submit.", results_df

    # 4. Prepare Submission 
    submission_data = {
        "username": username.strip(), 
        "agent_code": agent_code, 
        "answers": answers_payload
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
        
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
        
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
        
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Async Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  This app uses an advanced AsyncCodeAgentManager with timeout handling and fallback models.
        2.  Primary Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
        3.  Fallback Model: `Qwen/Qwen2.5-Coder-32B-Instruct` (used on timeout)
        4.  Log in to your Hugging Face account using the button below.
        5.  Click 'Run Evaluation & Submit All Answers' to process all questions efficiently.

        **Features:**
        - Async processing for better performance
        - Automatic fallback model on timeout
        - Batch processing of multiple questions
        - Enhanced error handling and logging
        - Timeout: 3 minutes per question

        ---
        **Note:** The async agent processes questions more efficiently and provides better error recovery.
        """
    )

    gr.LoginButton()

    # Add status display for agent information
    with gr.Row():
        with gr.Column():
            agent_info = gr.Markdown(
                """
                **Agent Configuration:**
                - Primary Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
                - Fallback Model: Qwen/Qwen2.5-Coder-32B-Instruct
                - Timeout: 180 seconds per question
                - Max Steps: 15
                """
            )

    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", 
        lines=8, 
        interactive=False,
        show_copy_button=True
    )
    
    results_table = gr.DataFrame(
        label="Questions and Agent Answers", 
        wrap=True,
        interactive=False
    )

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )


if __name__ == "__main__":
    print("\n" + "-"*30 + " Async App Starting " + "-"*30)
    
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" Async App Starting ")) + "\n")
    print("Launching Gradio Interface for Async Agent Evaluation...")
    
    # Log agent configuration
    print("Agent Configuration:")
    print("- Primary Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    print("- Fallback Model: Qwen/Qwen2.5-Coder-32B-Instruct")
    print("- Timeout: 180 seconds per question")
    print("- Async batch processing enabled")
    
    demo.launch(debug=True, share=False)