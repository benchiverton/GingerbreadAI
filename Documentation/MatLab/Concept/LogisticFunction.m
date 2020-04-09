function result = Logistic(A, B, X)
    result = 1 / (1 + exp(-B * (X - A)))
end