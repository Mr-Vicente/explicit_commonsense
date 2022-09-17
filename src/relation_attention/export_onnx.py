from pathlib import Path
from transformers.convert_graph_to_onnx import convert

if __name__ == '__main__':
    model_path = ''
    model_name = ''
    convert(framework="pt", model=model_path, output=Path(f"onnx/{model_name}.onnx"), opset=11)
