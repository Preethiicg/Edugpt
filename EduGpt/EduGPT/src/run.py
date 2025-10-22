import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import os
import time
import gradio as gr
from src.generating_syllabus import generate_syllabus
from src.teaching_agent import teaching_agent


# Load your OpenAI key from .env
with open(".env", "r") as f:
    env_file = f.readlines()
envs_dict = {key.strip("'"): value.strip("\n") for key, value in [i.split("=") for i in env_file]}
os.environ["OPENAI_API_KEY"] = envs_dict["OPENAI_API_KEY"]


def perform_task(input_text):
    """Generate a syllabus and seed the teaching agent."""
    task = "Generate a course syllabus to teach the topic: " + input_text
    syllabus = generate_syllabus(input_text, task)
    teaching_agent.seed_agent(syllabus, task)
    return syllabus


def user_message(user_message, history):
    """Handle user input."""
    if not user_message.strip():
        return "", history
    teaching_agent.human_step(user_message)
    history = history + [[user_message, None]]
    return "", history


def bot_message(history):
    """Generate the instructor's reply."""
    bot_reply = teaching_agent.instructor_step()
    history[-1][1] = ""
    for character in bot_reply:
        history[-1][1] += character
        time.sleep(0.02)
        yield history


def clear_chat():
    """Clear the chat history."""
    teaching_agent.conversation_history = []
    return []


# Build the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎓 Your AI Instructor\nWelcome! Generate a custom syllabus and start learning interactively.")
    
    with gr.Tab("📘 Create Your Syllabus"):
        topic_input = gr.Textbox(label="Enter the topic you want to learn:", placeholder="e.g. Machine Learning")
        syllabus_output = gr.Textbox(label="Generated Syllabus", lines=10)
        generate_button = gr.Button("✨ Generate Syllabus")
        generate_button.click(perform_task, topic_input, syllabus_output)

    with gr.Tab("🧑‍🏫 AI Instructor Chat"):
        chatbot = gr.Chatbot(height=400, label="Your Tutor")
        msg = gr.Textbox(label="Ask your tutor:", placeholder="Type your question here and press Enter...")
        clear = gr.Button("Clear Conversation")

        # Connect message submission
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_message, chatbot, chatbot
        )

        # Connect clear button
        clear.click(clear_chat, None, chatbot, queue=False)

# Launch app
demo.queue().launch(debug=True)
