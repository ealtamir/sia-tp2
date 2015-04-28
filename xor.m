source "multilayer.m"

function xorProblem()
    input_vec = [0 0; 0 1; 1 0; 1 1];
    expected = [0; 1; 1; 0];
    neurons = [2 1];
    weights = trainNeuronForXorProblem(input_vec, neurons, expected);
    for j = 1:length(input_vec)
        evalNeuron(input_vec(j, 1), weights, neurons)
    end
end

function weights = trainNeuronForXorProblem(input_vec, neurons, expected)
    % Each row represents a neuron
    % each column represents the weights spawning from 
    % a neuron or input.
    weights = rand(2, 3, 2);
    epochs = 1;
    trainNeuron(input_vec, expected, neurons, weights);
end
