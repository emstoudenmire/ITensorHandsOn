using ITensors, ITensorMPS

function initial_state(sites)
  psi = MPS(sites)
  n = length(psi)
  for j=1:n
    psi[j] = ITensor([1.,0],sites[j])
  end
  return psi
end
