source "multilayer.m"

clear partial_results;
global partial_results;

function approxProblem()
    global partial_results;
    range = 2 * pi;
    samples = 100;
    step = 2 * range / samples;
    input_vec = [-range:step:range](1:samples)';
    expected = analyticFunction(input_vec);
    neurons = [5 1];
    lrate = 0.1;
    act_func = @exponential;
    deriv_func = @deriv_exp;
    epochs = 10000;
    err_threshold = 0.01;
    val_threshold = 0.0001;

    [weights, epoch] = trainNetwork(input_vec, neurons, expected, act_func,
        deriv_func, lrate, epochs, err_threshold, val_threshold,
        @storeWeightsPartialResults, @analyticFunction);

    results = evalInput(input_vec, weights, neurons, act_func);
    plotAllResults(input_vec, results, partial_results, expected);
    gen_power = testGeneralizationPower(weights, neurons, act_func);
    printf("Completed %d epochs out of %d.\n", epoch, epochs);
    printf("Generalization error: %.2f percent\n", gen_power);
end

function gen_power = testGeneralizationPower(weights, neurons, act_func)
    samples = 10e3;
    test_samples = genTestSamples(samples)';
    real_vals = analyticFunction(test_samples)';
    approx_vals = evalInput(test_samples, weights, neurons, act_func);
    err = abs(real_vals - approx_vals) ./ real_vals;
    gen_power = (sum(err) / samples) * 100;
end

function storeWeightsPartialResults(input_vec, weights, neurons,
        act_func, iteration, epoch)
    global partial_results;
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

function val = analyticFunction(vector)
    val = 1 ./ (cos(vector) + 2);
end
