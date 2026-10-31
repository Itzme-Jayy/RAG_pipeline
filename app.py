import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question
from implementation.upload import save_uploaded_files
from implementation.ingest import ingest_documents
from implementation.clear_data import clear_all

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]

    try:
        answer, context = answer_question(last_message, prior)
        history.append({"role": "assistant", "content": answer})
        return history, format_context(context)

    except Exception as e:
        print("Error:", e)  # Shows actual error in terminal

        history.append({
            "role": "assistant",
            "content": "LLM is experiencing high demand. Please try again after some time."
        })

        return history, format_context(context)

def upload_documents(files):

    if not files:
        return "Please select documents first."

    # Save files permanently
    saved_paths = save_uploaded_files(files)

    # Display saved locations
    result = "Successfully uploaded!\n\n"

    # for path in saved_paths:
    #     result += f"📄 {path}\n"

    return result

def vectorize_documents():

    try:
        result = ingest_documents()
        return result

    except Exception as e:
        return f"Error during vectorization: {str(e)}"
    
def clear_data():
    result = clear_all()
    return (
            None,  # Clear file selection
            "",    # Clear upload status
            "",    # Clear vectorization status
            "*Retrieved context will appear here*",  # Reset context
            [],    # Clear chatbot
            "",    # Clear question textbox
            " --Cleared Everything" # Display success message
        )
def clear_uploaded_docs():
    return clear_all()
    

def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG application", theme=theme) as ui:

        gr.Markdown("## 📁 Upload Documents")


        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, type="messages", show_copy_button=True
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about uploaded docs",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )
            message.submit(
                put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
                ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                label="Select documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                type="filepath")

        
                upload_status = gr.Markdown()

                upload_button = gr.Button("Upload Documents")
                
                upload_button.click(
                fn=upload_documents,
                inputs=file_upload,
                outputs=upload_status)
            
            with gr.Column(scale=1):
                gr.Markdown("## Vector Database")

                vectorize_button = gr.Button(
                "Generate Embeddings and Store in ChromaDB",
                variant="primary")

                vectorize_status = gr.Textbox(
                    label="Vectorization Status",
                    interactive=False,
                    lines=5)

                vectorize_button.click(
                    fn=vectorize_documents,
                    inputs=[],
                    outputs=vectorize_status)
            
            
        with gr.Row():
            clear_button = gr.Button("Clear Everything")
            delete_docs = gr.Button("delete documents")
            output = gr.Markdown()
            delete_docs.click(clear_uploaded_docs, inputs=[], outputs=output)
            clear_button.click(
            fn=clear_data,
            inputs=[],
            outputs=[
                file_upload,
                upload_status,
                vectorize_status,
                context_markdown,
                chatbot,
                message,
                output])
    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
