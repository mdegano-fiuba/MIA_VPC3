import gradio as gr
from inference.predict import predict_image

def infer(img):
    return predict_image(img.name)

iface = gr.Interface(fn=infer, inputs="file", outputs="text")
iface.launch(server_name="0.0.0.0", server_port=7860)

