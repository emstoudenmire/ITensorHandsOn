using ITensors

function make_random_gates(sites; num_layers=8)
  n = length(sites)
  gates = ITensor[]
  for l=1:num_layers÷2
    for j=1:n-1
      # Make a random unitary matrix q
      q,_ = qr(randn(4,4))
      # Make an ITensor with q as the data
      g = ITensor(Matrix(q),sites[j],sites[j+1],sites[j]',sites[j+1]')
      push!(gates,g)
    end
    for j=reverse(1:n-1)
      # Make a random unitary matrix q
      q,_ = qr(randn(4,4))
      # Make an ITensor with q as the data
      g = ITensor(Matrix(q),sites[j],sites[j+1],sites[j]',sites[j+1]')
      push!(gates,g)
    end
  end
  return gates
end
