"""
Gradio app for MentorFlow - Teacher-Student RL System
Deployed on Hugging Face Spaces with GPU support
"""

import gradio as gr
import sys
import os
import subprocess
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "teacher_agent_dev"))
sys.path.insert(0, str(Path(__file__).parent / "student_agent_dev"))

def run_comparison(iterations: int, seed: int, use_deterministic: bool, device: str, progress=gr.Progress(track_tqdm=True)):
    """
    Run strategy comparison with LM Student.
    
    Args:
        iterations: Number of training iterations
        seed: Random seed (ignored if deterministic)
        use_deterministic: Use fixed seed=42
        device: 'cpu' or 'cuda' (GPU)
        progress: Gradio progress tracker
    """
    
    # Set device environment variable for subprocess
    # Check if CUDA is actually available before using
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
                if progress:
                    progress(0.0, desc="⚠️ GPU not available, using CPU...")
        except ImportError:
            device = "cpu"
            if progress:
                progress(0.0, desc="⚠️ PyTorch not available, using CPU...")
        except Exception:
            device = "cpu"
    
    # Set environment variable for subprocess to pick up
    os.environ["CUDA_DEVICE"] = device
    
    # Prepare command
    cmd = [
        sys.executable,
        "teacher_agent_dev/compare_strategies.py",
        "--iterations", str(iterations),
    ]
    
    if use_deterministic:
        cmd.append("--deterministic")
    else:
        cmd.extend(["--seed", str(int(seed))])
    
    try:
        if progress:
            progress(0.1, desc="Starting comparison...")
        
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
        
        if progress:
            progress(0.9, desc="Processing results...")
        
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
            if progress:
                progress(1.0, desc="Complete!")
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
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"✅ GPU Available: {torch.cuda.get_device_name(0)}"
        else:
            return "⚠️ No GPU available, using CPU"
    except:
        return "⚠️ Could not check GPU status"


# Create Gradio interface
with gr.Blocks(title="MentorFlow - Strategy Comparison") as demo:
    gr.Markdown("""
    # 🎓 MentorFlow - Teacher-Student RL System
    
    Compare three training strategies using LM Student (DistilBERT):
    1. **Random Strategy**: Random questions until student can pass difficult questions
    2. **Progressive Strategy**: Easy → Medium → Hard within each family
    3. **Teacher Strategy**: RL teacher agent learns optimal curriculum
    
    ## Usage
    
    1. Set parameters below
    2. Click "Run Comparison" to start training
    3. View results and generated plots
    
    **Note**: With LM Student, this will take 15-30 minutes for 500 iterations.
    """)
    
    # GPU Status
    with gr.Row():
        gpu_status = gr.Textbox(label="GPU Status", value=check_gpu(), interactive=False)
        refresh_btn = gr.Button("🔄 Refresh GPU Status")
    
    refresh_btn.click(fn=check_gpu, outputs=gpu_status)
    
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
            
            device = gr.Radio(
                choices=["cuda", "cpu"],
                value="cuda",
                label="Device",
                info="Use GPU (cuda) if available, CPU otherwise"
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
        inputs=[iterations, seed, use_deterministic, device],
        outputs=[output_text, output_plot]
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
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

