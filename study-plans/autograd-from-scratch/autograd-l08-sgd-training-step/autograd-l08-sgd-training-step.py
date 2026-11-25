import torch

def sgd_training_step(inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, learning_rate: float) -> tuple:
    """
    Returns old loss, new loss, updated weights, updated bias, weight gradients, and bias gradient.
    """
    learning_rate = float(learning_rate)

    x = inputs

    y = targets

    trained_weights = weights.clone()

    trained_biases = bias.clone()

    predictions  = torch.tanh(x@trained_weights + trained_biases)

    old_loss = ((predictions - y) ** 2).sum()

    preactivation_gradients = (2 * (predictions - y))*(1-predictions* predictions)

    weight_gradients = x.transpose(0, 1) @ preactivation_gradients
    
    bias_gradient = preactivation_gradients.sum()

    updated_weights = trained_weights - learning_rate * weight_gradients

    updated_bias = trained_biases - learning_rate * bias_gradient

    new_predictions = torch.tanh(x @ updated_weights + updated_bias)

    new_loss = ((new_predictions - y) ** 2).sum()

    values = (old_loss, new_loss, updated_weights, updated_bias, weight_gradients, bias_gradient)

    return tuple((value.clone() for value in values))


        

    
