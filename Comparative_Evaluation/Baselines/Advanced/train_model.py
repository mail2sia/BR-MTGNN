from common import base_arg_parser, load_data, resolve_device, set_seed, train_model
from models import AGCRNModel, DCRNNModel, PatchTSTModel, TFTModel, TimesFMModel


def build_model(name, args, data, device):
    n = data.num_nodes
    if name == "dcrnn":
        return DCRNNModel(n, args.seq_in_len, args.seq_out_len, data.adj).to(device)
    if name == "agcrn":
        return AGCRNModel(n, args.seq_in_len, args.seq_out_len).to(device)
    if name == "patchtst":
        return PatchTSTModel(n, args.seq_in_len, args.seq_out_len).to(device)
    if name == "tft":
        return TFTModel(n, args.seq_in_len, args.seq_out_len).to(device)
    if name == "timesfm":
        model = TimesFMModel(n, args.seq_in_len, args.seq_out_len).to(device)
        model.load_pretrained(device)
        return model
    raise ValueError(f"Unknown model '{name}'")


def main(model_name: str):
    parser = base_arg_parser(model_name)
    args = parser.parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    data = load_data(args, device)
    model = build_model(model_name, args, data, device)
    train_model(args, model, data, device)


if __name__ == "__main__":
    raise RuntimeError("Use model-specific entry scripts")
