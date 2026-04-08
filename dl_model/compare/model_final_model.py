from dl_model.final_model.model import TDNNPredictor

MODEL_NAME = "final_model"


def build_model(args):
    weight_path = getattr(args, "final_model_weight_path", "dl_model/final_model/sincnet_best_acc.pth")
    predictor = TDNNPredictor(device="cpu", weight_path=weight_path)
    return predictor.model
