source "multilayer.m"

global partial_results;

function approxProblem()
    global partial_results;
    range = 2 * pi;
    step = 1;
    input_vec = [-range:step:range]';
    expected = 1 ./ (cos(input_vec) + 2);
    neurons = [5 1];
    lrate = 0.3;
    act_func = @exponential;
    deriv_func = @deriv_exp;
    epochs = 10000;
    weights = trainNetwork(input_vec, neurons, expected, act_func,
        deriv_func, lrate, epochs, @storeWeightsPartialResults);
    results = evalInput(input_vec, weights, neurons, act_func);
    plotAllResults(input_vec, results, partial_results);
end

function storeWeightsPartialResults(input_vec, weights, neurons, act_func, iteration)
    global partial_results;
    partial_results(iteration, :) = evalInput(input_vec, weights, neurons, act_func);
end

function plotAllResults(input_vec, results, partial_results)
    plots = size(partial_results, 1) + 1
    for j = 1:size(partial_results, 1)
        subplot(plots, 1, j);
        plot(input_vec, partial_results(j, :));
    end
    subplot(plots, 1, plots);
    plot(input_vec, results);
end
