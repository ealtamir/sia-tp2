source "multilayer.m"

function xorProblem()
    input_vec = [0 0; 0 1; 1 0; 1 1];
    expected = [0; 1; 1; 0];
    neurons = [2 1];
    lrate = 0.6;
    act_func = @exponential;
    deriv_func = @deriv_exp;
    weights = trainNeuronForXorProblem(input_vec, neurons,
        expected, act_func, deriv_func, lrate);
    for j = 1:length(input_vec)
        evalNeuron(input_vec(j, :), weights, neurons)
    end
end

function weights = trainNeuronForXorProblem(input_vec, neurons, expected,
        act_func, deriv_func, lrate)
    % Each row represents a neuron
    % each column represents the weights spawning from 
    % a neuron or input.
    input_size = size(input_vec, 2) + 1;
    for j = 1:length(neurons)
        weights{j} = rand(neurons(j), input_size);
        input_size = neurons(j) + 1;
    end
    epochs = 2000;
    params = buildParamsCell(input_vec, expected, neurons,
        weights, epochs, act_func, deriv_func, lrate);
    weights = trainNeuron(params);
end
