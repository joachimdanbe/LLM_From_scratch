from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import gradio as gr

prompt_template_str = """
Your task is to explain the concept of {concept} to me in a way that is clear, concise (under 100 words), and tailored specifically to me. I am a data science student with a mathematical sciences background, working on AI-assisted industrial bearing fault diagnosis using vibration signals and LLMs. Use this context to make your explanation immediately relevant — connecting concepts to signal processing, machine learning, or industrial diagnostics where it genuinely helps. Keep personalization subtle and natural.
I want you to specify what doea each metric do and what is the Formula used to get It.
"""

prompt_template = PromptTemplate.from_template(prompt_template_str)

input_text = "Flesch RE"

prompt = prompt_template.format(concept = input_text)

print(prompt)

model = init_chat_model("gpt-4o-mini", model_provider = "openai")

response = model.invoke(prompt)

print(response.text)

def generate_explanation(input_text):
  prompt = prompt_template.format(concept = input_text)
  response = model.invoke(prompt)

  return response.text
print(generate_explanation("SMOG"))  


demo = gr.Interface(
  fn = generate_explanation,
  inputs=[gr.Textbox(label = "Concept",lines = 1)],
  outputs = [gr.Textbox(label = "Explanation", lines = 5)],
  flagging_mode = "never",
  title = "Metrics Readability Explainer",
)
#demo.launch()
  