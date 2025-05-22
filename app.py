from src.utils import datadir
import keras
import pandas as pd
from src.utils import draw_samples

model_path = datadir("data/model_v4_prod_v1")
print("PATH:", model_path)

model = keras.models.load_model(model_path)

input_dataset = pd.read_pickle(datadir("input_rot.pkl"))



draw_samples(model=model, input_dataset=input_dataset)


