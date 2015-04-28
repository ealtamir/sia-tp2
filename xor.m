source "multilayer.m"

function xorProblem()
    input_vec = [0 0; 0 1; 1 0; 1 1];
    expected = [0; 1; 1; 0];
    weights = trainNeuronForXorProblem(input_vec, expected);
    for j = 1:length(input_vec)
        evalNeuron(input_vec(j, 1), weights)
    end
end

function weights = trainNeuronForXorProblem(input_vec, expected)
    weights = rand(2, 3, 2);
    neurons = [2 1];
    epochs = 100;
    trainNeuron(input_vec, expected, neurons, weights);
end
