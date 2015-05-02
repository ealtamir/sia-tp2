global VALIDATION_ERROR_SAMPLES = 100;
global VALIDATION_ERROR_STEP = 10;
global SAMPLES = 4;
global EPOCH_ERROR = true;
global VALIDATION_ERROR = true;
global steps = 0;

function ret = adapt_lrate(lrate, prevErr, currErr, b=0.04, a=0.01)
    global steps;

    delta_Error = quadraticMeanError(currErr) - quadraticMeanError(prevErr);

    ret = lrate;
    if delta_Error > 0
        ret = ret -b*lrate;
        steps = 0;
    elseif delta_Error < 0
        steps += 1;
        if steps >= 3
            ret = ret + a;
        end
    end
end

function qme = quadraticMeanError(err)
    qme = sum(err.^2) / length(err);
end

function [weights, total_epochs] = trainNetwork(input_vec, neurons, expected, act_func,
    deriv_func, lrate, epochs, err_threshold, val_threshold, partialResultsFunc=false,
    gen_test)
    % Each row represents a neuron
    % each column represents the weights spawning from
    % a neuron or input.
    global SAMPLES;
    input_size = size(input_vec, 2) + 1;
    for j = 1:length(neurons)
        weights{j} = rand(neurons(j), input_size);
        input_size = neurons(j) + 1;
    end

    iter_epochs = round(epochs / SAMPLES);
    j = 1;
    proceed = true;
    total_epochs = 0;
    while (j <= SAMPLES && proceed)
        if is_function_handle(partialResultsFunc)
            partialResultsFunc(input_vec, weights, neurons, act_func, j, total_epochs);
        end

        [proceed, weights, epoch] = train(input_vec, expected, neurons, weights, iter_epochs,
            act_func, deriv_func, lrate, gen_test, err_threshold, val_threshold);
        j += 1;
        total_epochs += epoch;
    end
end

function results = evalInput(input_vec, weights, neurons, g)
    for j = 1:length(input_vec)
        results(1, j) = evalNeuron(input_vec(j, :), weights, neurons, g);
    end
end

function output = evalNeuron(input, weights, neurons, g)
    layers = length(neurons);
    new_input = [-1, input]';
    for layer = 1:layers
        output = g(weights{1, layer} * new_input);
        new_input = [-1; output];
    end
end

function [proceed, weights, epoch] = train(input_vec, expected, neurons, weights, epochs,
        act_func, deriv_func, lrate, gen_test, err_threshold=0.01, val_threshold=0.01)
    global VALIDATION_ERROR;
    global EPOCH_ERROR;
    val_err = avg_err = 0;
    epoch_size = size(input_vec, 1);
    proceed = true;
    epoch = 1;
    prevErr = zeros(1, size(expected, 2));
    while epoch < epochs && proceed
        shuffled_indexes = randperm(epoch_size);
        for j = 1:epoch_size
            % input matrixes have the input vector as a row, it must be transformed
            % to a column vector first.
            input = input_vec(shuffled_indexes(j), :)';
            [output, fields] = forward(input, neurons, weights, act_func);
            err(j) = expected(shuffled_indexes(j), :) - output;
            weights = backward(err(j), fields, input, neurons, weights,
                act_func, deriv_func, lrate);
        end
        %lrate = adapt_lrate(lrate, prevErr, err);
        %prevErr = err;

        if VALIDATION_ERROR && EPOCH_ERROR
            [error_test_passes, avg_err] = calcCuadraticMeanError(err, err_threshold);
            [val_test_passes, val_err] = performValidationTest(gen_test, val_threshold,
                weights, neuron, act_func);
            proceed = error_test_passes && val_test_passes;
        elseif VALIDATION_ERROR
            [proceed, val_err] = performValidationTest(gen_test, val_threshold,
                weights, neuron, act_func);
        elseif EPOCH_ERROR
            [proceed, avg_err] = calcCuadraticMeanError(err, err_threshold);
        end
        if proceed == false
            printf("Avg. epoch error: %.5f - Value error: %.5f.\n", avg_err, val_err);
        end

        epoch += 1;
    end
end
function [proceed, val_err] = performValidationTest(gen_test, val_threshold, weights, neuron, g)
    global VALIDATION_ERROR_SAMPLES;
    persistent CALLS = 10e6;
    proceed = true;
    if CALLS > 0 && mod(CALLS, VALIDATION_ERROR_STEP) == 0
        samples = genTestSamples(VALIDATION_ERROR_STEP);
        real_values = gen_test(samples);
        approx_values = evalInput(samples, weights, neurons, g);
        err = real_values - approx_values;
        [proceed, val_err] = calcCuadraticMeanError(err, val_threshold);
    elseif CALLS < 0
        CALLS += 10e6;
    end
    CALLS -= 1;
end

function [proceed, avg_err] = calcCuadraticMeanError(err_vec, err_threshold)
    proceed = true;
    avg_err = sum(err_vec .^ 2) / (2 * length(err_vec));
    if avg_err <= err_threshold
        proceed = false;
    end
end

function [output, fields] = forward(input, neurons, weights, g)
    layers = length(neurons);
    input = [-1; input];
    for layer = 1:layers
        v = weights{1, layer} * input;
        fields{1, layer} = v;
        output = g(v);
        input = [-1; output];
    end
end

function weights = backward(err, fields, input, neurons, weights, g, gderiv, lrate)
    layers = length(neurons);

    out_gradient = err * gderiv(fields{1, layers})';
    y = [-1; g(fields{layers - 1})]'; % row vector
    delta{1, layers} = lrate * out_gradient * y;

    gradients = out_gradient;
    for layer = (layers - 1) : -1 : 1
        suma = gradients * weights{layer + 1}(:, 2:end); % esto puede dar problemas si hay varias capas ocultas.
        deriv = gderiv(fields{layer})';
        gradients =  suma .* deriv;
        if layer - 1 > 0 % if not in input layer yet
            y = [-1; g(fields{layer - 1})];
        else
            y = [-1; input];
        end
        deltas = lrate * y * gradients;
        delta{1, layer} = deltas';
    end

    for j = 1:length(neurons)
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

function samples = genTestSamples(amount)
    samples = (4 * pi) * rand(1, amount) - (2 * pi);
end
