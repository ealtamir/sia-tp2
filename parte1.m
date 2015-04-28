input = [1 1; 1 -1; -1 1; -1 -1];
s = [1; -1; -1; -1];
weights = rand(3, 1);

function weights = train_network(input, s, weights, bias=0.4, b=1, g=@tanh)
    repeat = 100;
    iters = size(input, 1);
    input_size = size(input, 2);

    for j = 1:repeat
        for i = 1:iters
            a = [-1, input(i, :)];
            potential = sum(transpose(weights) .* a);
            output = g(b * potential);
            for n = 1:(input_size + 1)
                if n == 1
                    weights(1, 1) = weights(1, 1) + bias * (s(i, 1) - output) * (-1);
                else
                    weights(n, 1) = weights(n, 1) + bias * (s(i, 1) - output) * input(i, n - 1);
                end
            end
        end
    end
end

function result = test_network(input, weights, g=@tanh, b=1)
    result = g(b * sum(transpose(weights) .* [-1, input]));
end

weights = train_network(input, s, weights)
