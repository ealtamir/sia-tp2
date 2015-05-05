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
    neurons = [10 5 1];
    %neurons = [5 3 1];
    lrate = 0.3;
    act_func = @exponential;
    deriv_func = @deriv_exp;
    epochs = 2000;
    err_threshold = 0.03;
    val_threshold = 0.0001;

    alpha = 0;
    use_adapt_lrate = true;
    a = 0.05;
    b = 0.15;
    [weights, epoch] = trainNetwork(input_vec, neurons, expected, act_func,
        deriv_func, lrate, epochs, err_threshold, val_threshold,
        @storeWeightsPartialResults, @analyticFunction, use_adapt_lrate, a, b, alpha);

    results = evalInput(input_vec, weights, neurons, act_func);
    filename = buildFilename(samples, neurons, "exp", (alpha != 0), alpha, use_adapt_lrate, a, b)
    plotAllResults(input_vec, results, partial_results, expected, filename);
    gen_power = testGeneralizationPower(weights, neurons, act_func);
    printf("Completed %d epochs out of %d.\n", epoch, epochs);
    printf("Generalization error: %.2f percent\n", gen_power);
    fflush(stdout);
end

function gen_power = testGeneralizationPower(weights, neurons, act_func)
    samples = 10e3;
    total_error = 1 - 1/3;
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

function filename = buildFilename(samples, arch, act_fun, use_momentum=0, alfa=0, use_adapt_lrate=0, a=0, b=0)
    filename = strcat(act_fun, "_");
    if use_adapt_lrate
        filename = strcat(filename, "adapt_b", mat2str(b), "a", mat2str(a));
    end
    if use_momentum
        filename = strcat(filename, "momentum", mat2str(alfa));
    end
    filename = strcat(filename, "sample", mat2str(samples), "arch", mat2str(arch), ".png");
end

function plotAllResults(input_vec, results, partial_results, expected, filename) 
    plots = size(partial_results, 1) + 1;
    len = length(input_vec);
    for j = 1:size(partial_results, 1)
        subplot(plots, 1, j);
        plot(input_vec, partial_results(j, 1:len), input_vec, expected);
    end
    subplot(plots, 1, plots);
    plot(input_vec, results, input_vec, expected);
    print(filename, "-dpng");
end

function val = analyticFunction(vector)
    val = 1 ./ (cos(vector) + 2);
end
