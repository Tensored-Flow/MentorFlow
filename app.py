"""
Gradio app for MentorFlow - Teacher-Student RL System
Deployed on Hugging Face Spaces with GPU support
"""

import sys
import os
import subprocess
from pathlib import Path

# Monkey-patch to fix Gradio schema generation bug BEFORE importing gradio
# Prevents TypeError: argument of type 'bool' is not iterable
def _patch_gradio_schema_bug():
    """Patch Gradio's buggy schema generation."""
    try:
        from gradio_client import utils as gradio_client_utils
        
        # Patch get_type - the main buggy function
        if hasattr(gradio_client_utils, 'get_type'):
            _original_get_type = gradio_client_utils.get_type
            
            def _patched_get_type(schema):
                """Handle bool schemas that cause the bug."""
                if isinstance(schema, bool):
                    return "bool"
                if schema is None:
                    return "Any"
                if not isinstance(schema, dict):
                    return "Any"
                try:
                    return _original_get_type(schema)
                except TypeError as e:
                    if "is not iterable" in str(e):
                        return "Any"
                    raise
            
            gradio_client_utils.get_type = _patched_get_type
    
        # Patch the wrapper function too
        if hasattr(gradio_client_utils, '_json_schema_to_python_type'):
            _original_json_to_type = gradio_client_utils._json_schema_to_python_type
            
            def _patched_json_to_type(schema, defs=None):
                """Catch errors in schema conversion."""
                try:
                    return _original_json_to_type(schema, defs)
                except (TypeError, AttributeError) as e:
                    if "is not iterable" in str(e):
                        return "Any"
                    raise
            
            gradio_client_utils._json_schema_to_python_type = _patched_json_to_type
    except (ImportError, AttributeError):
        pass

# Apply patch BEFORE importing gradio
_patch_gradio_schema_bug()

# Now import gradio (patch will be in effect)
import gradio as gr

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "teacher_agent_dev"))
sys.path.insert(0, str(Path(__file__).parent / "student_agent_dev"))

def run_comparison(
    iterations: int,
    seed: int,
    use_deterministic: bool,
    device: str,
    retention_constant: float,
    use_mock_student: bool,
    use_mock_generator: bool,
    resample_eval: bool,
):
    """
    Run strategy comparison with PPO Student wrapper.
    
    Args:
        iterations: Number of training iterations
        seed: Random seed (ignored if deterministic)
        use_deterministic: Use fixed seed=42
        device: 'cpu' or 'cuda' (GPU)
    """
    
    # Set device environment variable for subprocess
    # On Hugging Face Spaces with GPU, try to use CUDA
    if device == "cuda":
        try:
            import torch
            # Check if CUDA is available
            if torch.cuda.is_available():
                try:
                    # Try to get device name to verify GPU works
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_count = torch.cuda.device_count()
                    print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
                    # Keep device as "cuda"
                except Exception as e:
                    print(f"⚠️ GPU detection failed: {e}")
                    print("   Attempting to use CUDA anyway (may work)...")
                    # Don't fallback immediately - let it try
            else:
                print("⚠️ CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            print("⚠️ PyTorch not available, falling back to CPU")
            device = "cpu"
        except Exception as e:
            print(f"⚠️ GPU check error: {e}, falling back to CPU")
            device = "cpu"
    
    # Set environment variable for subprocess to pick up
    os.environ["CUDA_DEVICE"] = device
    print(f"🔧 Using device: {device}")
    
    # Prepare command
    cmd = [
        sys.executable,
        "teacher_agent_dev/compare_strategies.py",
        "--iterations", str(iterations),
    ]
    # Defaults for demos
    if use_mock_student:
        cmd.append("--use-mock-student")
    if use_mock_generator:
        cmd.append("--use-mock-generator")
    if resample_eval:
        cmd.append("--resample-eval")
    else:
        # If not resampling, explicitly disable medium auto-resample
        cmd.append("--medium-generator") if not use_mock_generator else None
    if not use_mock_generator:
        cmd.append("--medium-generator")
    
    if use_deterministic:
        cmd.append("--deterministic")
    else:
        cmd.extend(["--seed", str(int(seed))])
    
    # Pass retention constant via env for student models that honor memory_decay
    os.environ["RETENTION_CONSTANT"] = str(retention_constant)
    
    try:
        # Ensure environment variables are passed to subprocess
        env = os.environ.copy()
        env["CUDA_DEVICE"] = os.environ.get("CUDA_DEVICE", device)
        
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            env=env,  # Pass environment variables
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        stdout_text = result.stdout
        stderr_text = result.stderr
        
        # Combine outputs
        full_output = f"=== STDOUT ===\n{stdout_text}\n\n=== STDERR ===\n{stderr_text}"
        
        if result.returncode != 0:
            return f"❌ Error occurred:\n{full_output}", None
        
        # Find output plot (check multiple possible locations)
        plot_paths = [
            Path(__file__).parent / "teacher_agent_dev" / "comparison_all_strategies.png",
            Path(__file__).parent / "comparison_all_strategies.png",
            Path.cwd() / "teacher_agent_dev" / "comparison_all_strategies.png",
        ]
        
        plot_path = None
        for path in plot_paths:
            if path.exists():
                plot_path = path
                break
        
        if plot_path:
            return f"✅ Comparison complete!\n\n{stdout_text}", str(plot_path)
        else:
            # Return output even if plot not found (might still be useful)
            error_msg = f"⚠️ Plot not found at expected locations.\n"
            error_msg += f"Checked: {[str(p) for p in plot_paths]}\n\n"
            error_msg += f"Output:\n{full_output}"
            return error_msg, None
                
    except subprocess.TimeoutExpired:
        return "❌ Timeout: Comparison took longer than 1 hour", None
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None


def check_gpu():
    """Check if GPU is available on Hugging Face Spaces."""
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                cuda_version = torch.version.cuda
                return f"✅ GPU Available: {gpu_name} (Count: {gpu_count}, CUDA: {cuda_version})"
            except Exception as e:
                # GPU might be available but not immediately accessible
                return f"✅ GPU Detected (accessing: {str(e)[:50]}...)"
        else:
            # On Hugging Face Spaces, check environment
            if os.getenv("SPACE_ID"):
                # Check if GPU hardware is allocated
                hf_hardware = os.getenv("SPACE_HARDWARE", "unknown")
                if "gpu" in hf_hardware.lower() or "t4" in hf_hardware.lower() or "l4" in hf_hardware.lower():
                    return f"⚠️ GPU Hardware ({hf_hardware}) allocated but not accessible yet. Try running anyway."
                return f"⚠️ No GPU on this Space (hardware: {hf_hardware}). Please configure GPU tier."
            return "⚠️ No GPU available, will use CPU"
    except ImportError:
        return "⚠️ PyTorch not installed"
    except Exception as e:
        return f"⚠️ GPU check error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="MentorFlow - Strategy Comparison") as demo:
    gr.Markdown("""
    # 🎓 MentorFlow - Teacher-Student RL System
    
    Compare three training strategies using PPO Student (RL wrapper):
    1. **Random Strategy**: Random questions until student can pass difficult questions
    2. **Progressive Strategy**: Easy → Medium → Hard within each family
    3. **Teacher Strategy**: RL teacher agent learns optimal curriculum
    
    ## Usage
    
    1. Set parameters below
    2. Click "Run Comparison" to start training
    3. View results and generated plots
    
    **Note**: PPO Student runs faster than the LM student; 100-200 iterations are usually enough to see trends.
    
    ### Memory Retention (Forgetting)
    The student includes a memory decay model. The **retention constant** controls how quickly the student forgets:
    - Higher = slower forgetting (knowledge sticks longer)
    - Lower = faster forgetting (needs more review)
    This affects the mock/LM student paths that use `memory_decay`.
    
    ### Controls
    - **Mock Student**: Fastest; use this for quick plots.
    - **Mock Generator**: Use legacy mock task bank; default is the medium generator (STEM-heavy, larger bank).
    - **Resample Eval**: Resamples evaluation tasks each checkpoint to add variance and reduce overly linear curves.
    """)
    
    # GPU Status
    with gr.Row():
        gpu_status = gr.Textbox(label="GPU Status", value=check_gpu(), interactive=False)
        refresh_btn = gr.Button("🔄 Refresh GPU Status")
    
    refresh_btn.click(fn=check_gpu, outputs=gpu_status, api_name="check_gpu")
    
    # Parameters
    with gr.Row():
        with gr.Column():
            iterations = gr.Slider(
                minimum=50,
                maximum=500,
                value=100,
                step=50,
                label="Iterations",
                info="Number of training iterations (higher = longer runtime)"
            )
            
            seed = gr.Number(
                value=42,
                label="Random Seed",
                info="Seed for reproducibility (ignored if deterministic)"
            )
            
            use_deterministic = gr.Checkbox(
                value=True,
                label="Deterministic Mode",
                info="Use fixed seed=42 for reproducible results"
            )
            
            use_mock_student = gr.Checkbox(
                value=True,
                label="Use Mock Student (Fast)",
                info="Fastest path; uncheck to try PPO student"
            )
            
            use_mock_generator = gr.Checkbox(
                value=False,
                label="Use Mock Task Generator",
                info="Use legacy mock tasks; default is medium generator (larger STEM bank)"
            )
            
            resample_eval = gr.Checkbox(
                value=True,
                label="Resample Eval Sets",
                info="Adds variance to curves; recommended when using medium generator"
            )
            
            device = gr.Radio(
                choices=["cuda", "cpu"],
                value="cuda",  # Default to GPU for HF Spaces with Nvidia 4xL4
                label="Device",
                info="GPU (cuda) recommended for Nvidia 4xL4, CPU fallback available"
            )
            
            retention_constant = gr.Slider(
                minimum=10.0,
                maximum=150.0,
                value=80.0,
                step=5.0,
                label="Student Retention Constant",
                info="Higher = slower forgetting; lower = faster forgetting (applies to students using memory_decay)"
            )
        
        with gr.Column():
            run_btn = gr.Button("🚀 Run Comparison", variant="primary", size="lg")
    
    # Output
    with gr.Row():
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Output",
                lines=15,
                max_lines=30,
                interactive=False
            )
        
        with gr.Column(scale=1):
            output_plot = gr.Image(
                label="Comparison Plot",
                type="filepath",
                height=500
            )
    
    # Run comparison
        run_btn.click(
            fn=run_comparison,
            inputs=[iterations, seed, use_deterministic, device, retention_constant, use_mock_student, use_mock_generator, resample_eval],
            outputs=[output_text, output_plot],
            api_name="run_comparison"
        )
    
    gr.Markdown("""
    ## 📊 Understanding Results
    
    The comparison plot shows:
    - **Learning Curves**: How each strategy improves over time
    - **Difficult Question Performance**: Accuracy on hard questions
    - **Curriculum Diversity**: Topic coverage over time
    - **Learning Efficiency**: Iterations to reach target vs final performance
    
    The **Teacher Strategy** should ideally outperform Random and Progressive strategies.
    """)

if __name__ == "__main__":
    # For Hugging Face Spaces
    # Monkey-patch above should fix schema bug, but upgrade to Gradio 5.x is recommended
    demo.launch()
