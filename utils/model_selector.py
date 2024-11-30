from nets.SIGNL import W2VGCS_cls, W2VGCS_enc_trn


def select_enc_model(args):
    model_mapping = {
        "SIGNL": W2VGCS_enc_trn,
    }

    if args.model not in model_mapping:
        raise ValueError(
            f"Model '{args.model}' not recognized. Available models are: {list(model_mapping.keys())}"
        )

    model_class = model_mapping[args.model]
    return model_class(args)


def select_model(args):
    model_mapping = {
        "SIGNL": W2VGCS_cls,
    }

    if args.model not in model_mapping:
        raise ValueError(
            f"Model '{args.model}' not recognized. Available models are: {list(model_mapping.keys())}"
        )

    model_class = model_mapping[args.model]
    return model_class(args)
