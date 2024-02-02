import gradio as gr

DESCRIPTION = """
![img](https://studerendeonline.dk/images/dynamic/company/logo/109481)

This is group X's submission for the LLM Hackathon.

"""


def dronning_chat(message, history):
    """
    This is where your logic goes!
    For an example see: https://www.gradio.app/guides/creating-a-chatbot-fast
    
    """
    return "Gud bevare Danmark!"


gr.ChatInterface(dronning_chat, description=DESCRIPTION, title="twoday Kapacity EGN OpenAI hackathon").launch()