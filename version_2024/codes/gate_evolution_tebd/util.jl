using ITensors

default_cutoff() = 1E-12
default_maxdim() = 5000

function qubit_sites(n)
  return [Index(2,"Qubit,Site,j=$j") for j=1:n]
end

function initial_state(sites)
  psi = MPS(sites)
  n = length(psi)
  for j=1:n
    psi[j] = ITensor([1.,0],sites[j])
  end
  return psi
end
