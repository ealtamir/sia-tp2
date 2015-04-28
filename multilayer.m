
function test()
    weights = rand(2, 3, 3);
    limit = 2 * pi;
    step = 0.01;
    input_vec = [-limit : step : limit];
    expected = 1 ./ (cos(input_vec) + 2);
    neurons = [2, 2, 1];
    for j = 1:length(input_vec)
        [weights, output, fields] = forward(input_vec(j), weights, neurons, @tangenth);
    end
end

function [weights, output, fields] = forward(input, weights, neurons, activation_func)
    layers = size(weights, 3);
    input = [-1; input];
    for layer = 1:layers
        % input is a row vector
        % weights must have the weights in its columns
        width = length(input);
        height = neurons(layer);
        v = weights(1:height, 1:width, layer) * input;
        % has the induced local fields of each layer.
        fields(:, layer) = v;
        input = [-1; activation_func(v)];
    end
    output = input;
end

function weights = backward(input, err, weights, fields, deriv_func, act_func, lrate=0.1)
    layers = size(weights, 3);
    gradients = err * deriv_func(fields(:, layers));
    deltas = lrate * act_func(fields(:, layers - 1)) * gradients;
    weights(:, :, layers) = weights(:, :, layers) + deltas;
    for layer = (layers - 1) : -1 : 1
        gradients = deriv_func(fields(:, layer))' * (weights(:, :, layer + 1) * gradients');
        if layer - 1 > 0
            deltas = lrate * gradients * act_func(fields(:, layer - 1));
        else
            % use input because we're in the first layer
            deltas = lrate * gradients * [-1, input]';
        end
        weights(:, :, layer) = weights(:, :, layer) + deltas;
    end
end

function exponential(a, beta=1)
    return 1 / (1 + exp(-beta * a));
end

function tangenth(a, beta=1)
    return tanh(beta * a);
end
