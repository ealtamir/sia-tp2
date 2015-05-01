source "multilayer.m"

function approxProblem()
    global partial_results;
    range = 2 * pi;
    samples = 100;
    step = 2 * range / samples;
    input_vec = [-range:step:range](1:samples)';
    expected = 1 ./ (cos(input_vec) + 2);
    neurons = [5 1];
    lrate = 0.4;
    act_func = @exponential;
    deriv_func = @deriv_exp;
    epochs = 1000;
    weights = trainNetwork(input_vec, neurons, expected, act_func,
        deriv_func, lrate, epochs, @storeWeightsPartialResults);
    results = evalInput(input_vec, weights, neurons, act_func);
    plotAllResults(input_vec, results, partial_results, expected);
end

function storeWeightsPartialResults(input_vec, weights, neurons, act_func, iteration)
    global partial_results;
    %printf("updating partial results. Iteration %d\n", iteration);
    result = evalInput(input_vec, weights, neurons, act_func);
    partial_results(iteration, 1:length(result)) = result;
end

function plotAllResults(input_vec, results, partial_results, expected)
    plots = size(partial_results, 1) + 1;
    len = length(input_vec);
    for j = 1:size(partial_results, 1)
        subplot(plots, 1, j);
        plot(input_vec, partial_results(j, 1:len), input_vec, expected);
    end
    subplot(plots, 1, plots);
    plot(input_vec, results, input_vec, expected);
end
