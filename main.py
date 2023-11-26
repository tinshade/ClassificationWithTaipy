from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np

model = models.load_model("assets/baseline.keras")


class_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def predict_image(model, path_to_image):
    img = Image.open(path_to_image)
    img = img.convert("RGB").resize((32, 32))

    # Normalizing image
    data = np.asarray(img)
    print("Before: ", data[0][0])  # Printing color of very first pixel
    data = data / 255

    # Comparing stuff to see if we broke something
    print("After: ", data[0][0])  # Printing color of very first pixel

    # Tricking model into thinking it is looking at an array of sample images and not a single image
    probability = model.predict(np.array([data])[:1])
    probes = probability.max()
    prediction = class_names[np.argmax(probability)]
    return (probes, prediction)


image_path = "assets/placeholder_image.png"
prediction, prob, content = "", "", ""


image_control_component = """
<|text-center|
<|{"assets/logo.png"}|image|width=10vw|height=25vh|>

<|{content}|file_selector|extensions=.png|>

Select an image!


<|{prediction}|>



<|{image_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""


index = image_control_component


def on_change(state, variable_name, variable_value):
    if variable_name == "content":
        state.image_path = variable_value
        probes, prediction = predict_image(model, variable_value)
        state.prob = round(probes * 100)  # Converting decimal to percentage
        state.prediction = f"This is a : {prediction}"


app = Gui(page=index)
if __name__ == "__main__":
    app.run(use_reloader=True)
