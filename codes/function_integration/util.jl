
function evaluate_all(M::MPS; precision = length(M))
    #TODO: implement precision keyword argument
    s = siteinds(M)
    P1 = M[1] * onehot(s[1] => 1)
    P2 = M[1] * onehot(s[1] => 2)
    (length(M) == 1) && return [scalar(P1), scalar(P2)]
    L = MPS(M[2:end])
    R = MPS(M[2:end])
    L[1] *= P1
    R[1] *= P2
    return vcat(evaluate_all(L), evaluate_all(R))
end
