function test()
    weights = rand(2, 3, 3);
    limit = 2 * pi;
    step = 0.01;
    input_vec = [-limit : step : limit];
    expected = 1 ./ (cos(input_vec) + 2);
    neurons = [2, 2, 1];
    for j = 1:length(input_vec)
        [weights, output, fields] = forward(input_vec(j), weights, neurons, @tangenth);
    end
end
