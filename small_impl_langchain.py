from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import gradio as gr
# Model: gpt-4o-mini | Provider: openai
prompt_template_str = """
you ara an expert in Machine Learning, someone non from the domain want to understand {input_text} and also the mathematical formula behind.
- use simple word to explain to him
- be clear precis and consis

Avoid using others technical words in your explanation.
"""

promt_template = PromptTemplate.from_template(prompt_template_str)

prompt = promt_template.format(input_text= "accuracy")

#print(prompt)


model = init_chat_model("gpt-4o-mini", model_provider="openai")
response = model.invoke(prompt)
print(response.text)

def generate_explanation(input_text):
  
  prompt = promt_template.format(input_text=input_text)
  response = model.invoke(prompt)

  return response.text



demo = gr.Interface(
    fn=generate_explanation,
    inputs=[gr.Textbox(label="Enter an ML concept", lines=1)],
    outputs=[gr.Textbox(label="Explanation", lines=5)],
    flagging_mode="never",
    title="ML Metrics Explainer",
    description="Enter any ML metric and get a clear explanation with its formula."
)

# demo.launch()




