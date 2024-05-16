import torch
from sklearn.metrics import log_loss


def log_loss_metric(
    preds: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the log loss metric.

    Parameters
    ----------
    preds: torch.Tensor
        The model predictions.
    target: torch.Tensor
        The target labels.

    Returns
    -------
    torch.Tensor
        The log loss metric.
    """

    preds = preds.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    return torch.tensor(log_loss(target, preds))
