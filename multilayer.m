function output = evalNeuron(input, weights, neurons, g=@exponential)
    layers = length(neurons);
    new_input = [-1, input]';
    for layer = 1:layers
        output = g(weights{1, layer} * new_input);
        new_input = [-1; output];
    end
end

function weights = trainNeuron(input_vec, expected, neurons, weights, epochs=1)
    epoch_size = size(input_vec, 1);
    for epoch = 1:epochs
        for j = 1:epoch_size
            % input matrixes have the input vector as a row, it must be transformed
            % to a column vector first.
            input = input_vec(j, :)';
            [output, fields] = forward(input, weights, neurons, @exponential);
            err = expected(j, :) - output;
            weights = backward(input, err, weights, fields, neurons, @deriv_exp, @exponential);
        end
    end
end

function [output, fields] = forward(input, weights, neurons, activation_func)
    layers = length(neurons);
    input = [-1; input];
    for layer = 1:layers
        v = weights{layer} * input;

        % has the induced local fields of each layer.
        % v is column vector
        fields{layer} = v;
        output = activation_func(v);
        input = [-1; output];
    end
end

function weights = backward(input, err, weights, fields, neurons, deriv_func, act_func, lrate=0.6)
    layers = length(neurons);

    % delta = e_j * g'(v_j(n)), where v_j represents the sums at the output
    % gradients is a row vector
    gradients = err * deriv_func(fields{layers})';

    % outputs from previous layer of neurons
    % y is a column vector
    y = [-1; act_func(fields{layers - 1})]';

    d{1, layers} = w1 = lrate * gradients * y;

    % hasta acá da todo lo mismo
    for layer = (layers - 1) : -1 : 1
        pesos = weights{layer + 1}(:, 2:end);
        suma = (gradients * pesos);
        deriv = deriv_func(fields{layer})';
        gradients =  suma .* deriv;
        if layer - 1 > 0 % still not in first layer
            y = [-1; act_func(fields{layer - 1})]';
        else
            % use input because we're in the first layer
            y = [-1; input];
        end
        deltas = lrate * y * gradients;
        d{1, layer} = deltas';
    end
    % hasta acá todo lo mismo

    for j = 1:length(neurons)
        weights{1, j} += d{1, j};
    end
end

function output = exponential(a, beta=0.5)
    output = 1 ./ (1 + exp(-2 * beta * a));
end

function output = deriv_exp(pot, b=0.5)
    a = exp(2 * b * pot);
    output = (2 * b * a) ./ ((a + 1) .^ 2);
end

function output = tangenth(a, beta=1)
    output =  tanh(beta * a);
end

function output = deriv_tan(a, beta=1)
    output = beta * (1 - tanh(beta * a) .^ 2);
end
