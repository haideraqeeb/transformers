import torch

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out the all the values in the matrix if i <= j is true
    i < j if mask_diagonal is false

    This operation is an inplace operation
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Finds the device string either for the best available device, or for the device corresponding to the argument

    :return: afformentioned device string
    """

    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'