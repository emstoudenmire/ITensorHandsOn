using ITensors

function make_gates(sites)
  n = length(sites)
  gates = ITensor[]

  num_layers = 8
  for l=1:num_layers÷2
    for j=1:n-1
      q,_ = qr(randn(4,4))
      g = ITensor(Matrix(q),sites[j],sites[j+1],sites[j]',sites[j+1]')
      push!(gates,g)
    end
    for j=reverse(1:n-1)
      q,_ = qr(randn(4,4))
      g = ITensor(Matrix(q),sites[j],sites[j+1],sites[j]',sites[j+1]')
      push!(gates,g)
    end
  end

  return gates
end
