"""
Model utility functions
"""
import math
import torch


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    linear decay.

    Args:
        step: the current step
        total_steps: total number of steps
        warmup_steps: the number of warmup steps
    """
    step = min(step, total_steps)

    multiplier = (1 - (step - warmup_steps) / (total_steps - warmup_steps))

    if warmup_steps > 0 and step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        multiplier = warmup_percent_done

    return multiplier


def get_tsa_threshold(schedule: str, global_step: int, num_train_steps: int,
    start: float, end: float) -> float:
    r"""Get threshold for Training Signal Annealing. From the UDA paper:
    If the model’s predicted probability for the correct category pθ(y*|x) is higher than
    a threshold ηt, we remove that example from the loss function.
    Please see the paper for more details.

    Args:
        schedule: one of 'linear_schedule', 'exp_schedule', 'log_schedule'
        global_step: the current global step
        num_train_steps: the total number of training steps
        start: starting threshold
        end: ending threshold
    """
    training_progress = float(global_step) / float(num_train_steps)
    if schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = math.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - math.exp((-training_progress) * scale)
    return threshold * (end - start) + start
