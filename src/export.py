import pickle
import os

def export_best_model(model_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_dict["model"], f)
    return path