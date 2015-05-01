source "multilayer.m"

function xorProblem()
    input_vec = [0 0; 0 1; 1 0; 1 1];
    expected = [0; 1; 1; 0];
    neurons = [2 1];
    lrate = 0.1;
    act_func = @exponential;
    deriv_func = @deriv_exp;
    epochs = 10000;
    weights = trainNetwork(input_vec, neurons,
        expected, act_func, deriv_func, lrate, epochs);
    for j = 1:length(input_vec)
        evalNeuron(input_vec(j, :), weights, neurons)
    end
end
