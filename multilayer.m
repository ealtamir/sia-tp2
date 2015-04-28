function output = evalNeuron(input, weights, neurons, g=@tangenth)
    layers = size(weights, 3);
    new_input = [-1; input];
    for layer = 1:layers
        width = length(new_input);
        height = neurons(layer);
        output = g(weights(1:height, 1:width, layer) * new_input);
        new_input = [-1; output];
    end
end

function trainNeuron(input_vec, expected, neurons, weights, epochs=100)
    epoch_size = size(input_vec, 1);
    for epoch = 1:epochs
        for j = 1:epoch_size
            % input matrixes have the input vector as a row, it must be transformed
            % to a column vector first.
            input = input_vec(j, :)';
            [output, fields] = forward(input, weights, neurons, @tangenth);
            err = expected(j, :) - output;
            weights = backward(input, err, weights, fields, neurons, @deriv_tan, @tangenth);
        end
    end
end

function [output, fields] = forward(input, weights, neurons, activation_func)
    layers = size(weights, 3);
    input = [-1; input];
    for layer = 1:layers
        width = length(input);
        height = neurons(layer);
        v = weights(1:height, 1:width, layer) * input;

        % has the induced local fields of each layer.
        % v is column vector
        fields(:, layer) = v;
        output = activation_func(v);
        input = [-1; output];
    end
end

function weights = backward(input, err, weights, fields, neurons, deriv_func, act_func, lrate=0.1)
    layers = size(weights, 3);

    % delta = e_j * g'(v_j(n)), where v_j represents the sums at the output
    % gradients is a row vector
    gradients = err * deriv_func(fields(1:neurons(layers), layers))';

    % outputs from previous layer of neurons
    % y is a column vector
    y = act_func(fields(1:neurons(layers - 1), layers - 1));

    if length(gradients) == 1
        deltas = lrate * gradients * y';
    else
        deltas = lrate * gradients * y;
    end
    w = length(y) + 1
    h = neurons(layers)
    weights(1:h, 2:w, layers) = weights(1:h, 2:w, layers) + deltas;
    for layer = (layers - 1) : -1 : 1
        gradients = deriv_func(fields(1:neurons(layer), layer))' * sum(weights(1:h, 1:w, layer + 1) * gradients');
        gradients
        if layer - 1 > 0
            y = act_func(fields(1:neurons(layer - 1), layer - 1));
            deltas = lrate * gradients * y;
            w = length(y) + 1;
        else
            % use input because we're in the first layer
            deltas = lrate * gradients * [-1; input];
            w = length(input) + 1;
        end
        h = neurons(layer);
        % deltas
        deltas;
        weights(1:h, 2:w, layer) = weights(1:h, 2:w, layer) + deltas;
    end
end

function ouput = exponential(a, beta=1)
    output = 1 / (1 + exp(-beta * a));
end

function output = tangenth(a, beta=1)
    output =  tanh(beta * a);
end

function output = deriv_tan(a, beta=1)
    output = beta * (1 - tanh(beta * a) .^ 2);
end
