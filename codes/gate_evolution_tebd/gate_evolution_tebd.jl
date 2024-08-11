using ITensors
using ITensorMPS
using Random

include("make_gates.jl")

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

function apply_gate(g, psi; cutoff, maxdim)
  j = findsite(psi,inds(g))
  psi = orthogonalize(psi,j)
  gpsi = noprime(g*psi[j]*psi[j+1])
  u,s,v = svd(gpsi,uniqueinds(psi[j],psi[j+1]); cutoff,maxdim)
  psi[j] = u
  psi[j+1] = s*v
  return psi
end

function main(; n=40, cutoff=1E-12, maxdim=5000, seed=1)
  Random.seed!(seed)

  sites = qubit_sites(n)
  psi = initial_state(sites)

  gates = make_gates(sites)
  for g in gates
    psi = apply_gate(g, psi; cutoff, maxdim)
  end

  return psi
end
