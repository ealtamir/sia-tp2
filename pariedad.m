salidas = [0 1 1 0];
entradas = [0 0 -1; 0 1 -1; 1 0 -1; 1 1 -1];
%pesos = [rand(1, 2) 1; rand(1, 2) 1; rand(1, 2) 1];
pesos = rand(3, 3);


function pot = potencial (entrada, neurona, pesos)
  pot = sum(entrada .* pesos(neurona, :));
end

function ret = g (pot)
  b = 0.5;
  ret = 1/(1 + exp(-2*b*pot));
%  if ret < 0.1
%    ret = 0;
%  endif
%  if ret > 0.9
%    ret = 1;
%  endif
end

function ret = gderiv (pot)
  b = 0.5;
  ret = (2*b*exp(2*b*pot)) / ((exp(2*b*pot) + 1)**2);
end

n = 0.6;
for j = 1:2000
  for i = 1:4
    entradaOculta = [0 0 -1];
    potencialCapaOculta = [0 0];
    potencialCapaOculta(1) = potencial(entradas(i, :), 1, pesos);
    potencialCapaOculta(2) = potencial(entradas(i, :), 2, pesos);
    entradaOculta(1) = g(potencialCapaOculta(1));
    entradaOculta(2) = g(potencialCapaOculta(2));
    potencialCapaSalida = potencial(entradaOculta, 3, pesos);
    salida = g(potencialCapaSalida);
    err = (salidas(i) - salida);
    deltaCapaSalida = err*gderiv(potencialCapaSalida);
    gderivCapaOculta = [0 0];
    gderivCapaOculta(1) = gderiv(potencialCapaOculta(1));
    gderivCapaOculta(2) = gderiv(potencialCapaOculta(2));
    deltaCapaOculta = gderivCapaOculta .* (pesos(3, 1:2) * deltaCapaSalida);
    pesos(1, :) += (deltaCapaOculta(1) .* entradas(i, :)) * n;
    pesos(2, :) += (deltaCapaOculta(2) .* entradas(i, :)) * n;
    pesos(3, :) += (deltaCapaSalida .* entradaOculta) * n;
    %(deltaCapaOculta(1) .* entradas(i, :)) * n
    %(deltaCapaOculta(2) .* entradas(i, :)) * n
    %(deltaCapaSalida .* entradaOculta) * n
  end
end

pesos

for j = 1:4
    entradaOculta = [0 0 -1];
    potencialCapaOculta = [0 0];
    potencialCapaOculta(1) = potencial(entradas(j, :), 1, pesos);
    potencialCapaOculta(2) = potencial(entradas(j, :), 2, pesos);
    entradaOculta(1) = g(potencialCapaOculta(1));
    entradaOculta(2) = g(potencialCapaOculta(2));
    potencialCapaSalida = potencial(entradaOculta, 3, pesos);
    salida = g(potencialCapaSalida)
end



