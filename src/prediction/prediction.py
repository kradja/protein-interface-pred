from src.prediction import baseline_models_prediction, gnn_models_prediction


def execute(config):
    # input_settings
    input_settings = config["input_settings"]

    # output settings
    output_settings = config["output_settings"]

    # classification settings
    classification_settings = config["classification_settings"]
    type = classification_settings["type"]

    if type == "baseline":
        baseline_models_prediction.execute(input_settings, output_settings, classification_settings)
    elif type == "gnn":
        gnn_models_prediction.execute(input_settings, output_settings, classification_settings)