using ITensors, ITensorMPS

default_cutoff() = 1E-12
default_maxdim() = 5000

function qubit_sites(n)
  return [Index(2,"Qubit,Site,j=$j") for j=1:n]
end
