import os
import gradio as gr
import spaces

if not os.path.exists("vectorstore.index"):
    print("Vector database missing. Running ingestion...")
    import ingest
    ingest.run_ingest()

from main import app as custom_app

# The ZeroGPU supervisor requires a @spaces.GPU function tied to a Gradio event listener.
@spaces.GPU
def dummy_gpu_function():
    return "GPU is active"

# Create a dummy Gradio app that actually binds the GPU function so the supervisor detects it!
demo = gr.Blocks()
with demo:
    gr.Markdown("# Voice RAG Backend is Running!")
    btn = gr.Button("Wake up GPU (Internal)")
    out = gr.Textbox()
    btn.click(dummy_gpu_function, inputs=[], outputs=[out])

# Mount the dummy Gradio app onto our existing FastAPI app.
app = gr.mount_gradio_app(custom_app, demo, path="/gradio")

# Run Uvicorn directly to bypass HF's Gradio runner and serve our custom FastAPI app.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
