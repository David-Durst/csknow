import torch

def models_equal(model_a: torch.nn.Module, model_b: torch.nn.Module, script_modules: bool) -> bool:
    all_equal = True
    for p1, p2 in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        if not torch.equal(p1[1], p2[1]):
            all_equal = False
            break

    if not all_equal:
        return False

    if script_modules:
        model_a_code_constants = model_a.code_with_constants
        model_b_code_constants = model_b.code_with_constants
        if model_a_code_constants[0] != model_b_code_constants[0]:
            print("code mismatch")
            return False

        model_a_constants = model_a_code_constants[1].const_mapping
        model_b_constants = model_b_code_constants[1].const_mapping
        if len(model_a_constants.keys()) != len(model_b_constants.keys()):
            print("keys different lengths")
            return False
        for k in model_a_constants.keys():
            if not torch.equal(model_a_constants[k], model_b_constants[k]):
                print(f"{k} difference: {model_a_constants[k]}, {model_b_constants[k]}")
                return False

    return True

def script_non_script_models_equal(script_model: torch.nn.Module, non_script_model: torch.nn.Module,
                                   trace_tensor: torch.Tensor) -> bool:
    new_script_model = torch.jit.trace(script_model.to("cpu"), trace_tensor)
    return models_equal(script_model, new_script_model, True)

