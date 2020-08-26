
# def valid_quant_model(model):
#     if not model.torch_model_exists() or "quantize" not in model.__str__():
#         return False
#     return True

# def valid_model(model):
#     if not model.torch_model_exists() :
#         return False
#     return True

# debug
def valid_quant_model(model):
    if "quantize" not in model.__str__():
        return False
    return True

def valid_model(model):
    # if not model.torch_model_exists() :
    #     return False
    return True