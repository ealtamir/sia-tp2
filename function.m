source "multilayer.m"

clear partial_results;
global partial_results;

function approxProblem()
    global partial_results;
    range = 2 * pi;
    samples = 200;
    step = 2 * range / samples;
    input_vec = [-range:step:range](1:samples)';
    expected = analyticFunction(input_vec);
    neurons = [5 1];
    lrate = 0.3;
    act_func = @tangenth;
    deriv_func = @deriv_tan;
    epochs = 2000;
    err_threshold = 0.001;
    val_threshold = 0.0001;

    use_momentum = false;
    alfa = 0;
    use_adapt_lrate = true;
    %a = 0.01;
    %b = 0.05;
    for a = 0.01:0.05:1
        for b = 0.01:0.05:1
            [weights, epoch] = trainNetwork(input_vec, neurons, expected, act_func,
                deriv_func, lrate, epochs, err_threshold, val_threshold,
                @storeWeightsPartialResults, @analyticFunction, use_adapt_lrate);

            results = evalInput(input_vec, weights, neurons, act_func);
            filename = buildFilename(samples, neurons, "tanh", use_momentum, alfa, use_adapt_lrate, b, a)
            plotAllResults(input_vec, results, partial_results, expected, filename);
            gen_power = testGeneralizationPower(weights, neurons, act_func);
            printf("Completed %d epochs out of %d.\n", epoch, epochs);
            printf("Generalization error: %.2f percent\n", gen_power);
            fflush(stdout);
        end
    end
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

function filename = buildFilename(samples, arch, act_fun, use_momentum=0, alfa=0, use_adapt_lrate=0, b=0, a=0)
    filename = strcat(act_fun, "_");
    if use_adapt_lrate
        filename = strcat(filename, "adapt_b", mat2str(b), "a", mat2str(a));
    end
    if use_momentum
        filename = strcat(filename, "momalfa", mat2str(alfa));
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
