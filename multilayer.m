global INPUT = 1
global EXPECTED_OUTPUT = 2
global NEURONS = 3
global WEIGHTS = 4
global EPOCHS = 5
global ACT_FUNC = 6
global DERIV_FUNC = 7
global DERIV_FUNC = 8
global LRATE = 8


function output = evalNeuron(input, weights, neurons, g=@exponential)
    layers = length(neurons);
    new_input = [-1, input]';
    for layer = 1:layers
        output = g(weights{1, layer} * new_input);
        new_input = [-1; output];
    end
end

function weights = trainNeuron(params)
    input_vec = params{INPUT};
    expected = params{EXPECTED_OUTPUT};
    epoch_size = size(input_vec, 1);
    for epoch = 1:params{EPOCHS}
        for j = 1:epoch_size
            % input matrixes have the input vector as a row, it must be transformed
            % to a column vector first.
            input = input_vec(j, :)';
            [output, fields] = forward(input, params);
            err = expected(j, :) - output;
            weights = backward(err, fields, input, params);
        end
    end
end

function [output, fields] = forward(input, params)
    layers = length(params{NEURONS});
    weights = params{WEIGHTS};
    g = params{ACT_FUNC};

    input = [-1; input];
    for layer = 1:layers
        v = weights{1, layer} * input;
        fields{1, layer} = v;
        output = g(v);
        input = [-1; output];
    end
end

function weights = backward(err, fields, input, params)
    layers = length(params{NEURONS});
    g = params{ACT_FUNC};
    gderiv = params{DERIV_FUNC};

    out_gradient = err * deriv_func(fields{layers})';
    y = [-1; g(fields{layers - 1})]'; % row vector
    delta{1, layers} = lrate * out_gradient * y;

    gradients = out_gradient;
    for layer = (layers - 1) : -1 : 1
        suma = gradients * weights{layer + 1}(:, 2:end); % esto puede dar problemas si hay varias capas ocultas.
        deriv = gderiv(fields{layer})';
        gradients =  suma .* deriv;
        if layer - 1 > 0 % if not in input layer yet
            y = [-1; g(fields{layer - 1})]';
        else
            y = [-1; input];
        end
        deltas = params{LRATE} * y * gradients;
        delta{1, layer} = deltas';
    end

    for j = 1:length(params{NEURONS})
        weights{1, j} += delta{1, j};
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

function params = buildParamsCell(input_vec, expected, neurons, weights, epochs,
        g, gderiv, lrate)
    global INPUT
    global EXPECTED_OUTPUT
    global NEURONS
    global WEIGHTS
    global EPOCHS
    global ACT_FUNC
    global DERIV_FUNC
    global DERIV_FUNC
    global LRATE

    params{INPUT} = input_vec;
    params{EXPECTED_OUTPUT} = expected;
    params{NEURONS} = neurons;
    params{WEIGHTS} = weights;
    params{EPOCHS} = epochs;
    params{ACT_FUNC} = g;
    params{DERIV_FUNC} = gderiv;
    params{LRATE} = lrate;
end
