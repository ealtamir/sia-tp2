source "multilayer.m"

function xorProblem()
    input_vec = [0 0; 0 1; 1 0; 1 1];
    expected = [0; 1; 1; 0];
    neurons = [2 1];
    weights = trainNeuronForXorProblem(input_vec, neurons, expected);
    weights
    for j = 1:length(input_vec)
        evalNeuron(input_vec(j, :), weights, neurons)
    end
end

function weights = trainNeuronForXorProblem(input_vec, neurons, expected)
    % Each row represents a neuron
    % each column represents the weights spawning from 
    % a neuron or input.
    input_size = size(input_vec, 2) + 1;
    for j = 1:length(neurons)
        weights{j} = rand(neurons(j), input_size);
        input_size = neurons(j) + 1;
    end
    epochs = 2000;
    weights = trainNeuron(input_vec, expected, neurons, weights, epochs);
    %trainNeuron(input_vec, expected, neurons, weights);
end
