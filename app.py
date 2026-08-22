import os
import gradio as gr

if not os.path.exists("vectorstore.index"):
    print("Vector database missing. Running ingestion...")
    import ingest
    ingest.run_ingest()

from main import app as custom_app

# Create a dummy Gradio app to satisfy Hugging Face's SDK requirement
demo = gr.Blocks()
with demo:
    gr.Markdown("# Voice RAG Backend is Running!")
    gr.Markdown("The FastAPI application is successfully being hosted via the Gradio SDK.")

# Mount the dummy Gradio app onto our existing FastAPI app.
# We mount it at /gradio so it doesn't interfere with our root (/) index.html frontend!
app = gr.mount_gradio_app(custom_app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
