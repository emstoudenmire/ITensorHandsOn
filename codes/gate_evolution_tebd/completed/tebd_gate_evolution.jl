using ITensors
using ITensorMPS
using Random

include("util.jl")
include("make_gates.jl")

function apply_gate(g, psi; cutoff, maxdim)
  j = findsite(psi,inds(g))  # the gate acts on sites j and j+1
  psi = orthogonalize(psi,j) # this is a technical step to 
                             # ensure truncation accuracy
  gpsi = noprime(g*psi[j]*psi[j+1])
  u,s,v = svd(gpsi,uniqueinds(psi[j],psi[j+1]); cutoff,maxdim)
  psi[j] = u
  psi[j+1] = s*v
  psi[j+1] /= norm(psi[j+1])
  return psi
end

function tebd_gate_evolution(gates, psi; cutoff=default_cutoff(), maxdim=default_maxdim(), seed=1)
  for g in gates
    psi = apply_gate(g, psi; cutoff, maxdim)
  end
  return psi
end

function main(; n=40, cutoff=default_cutoff(), maxdim=default_maxdim(), seed=1)
  Random.seed!(seed)

  if isnothing(sites)
    sites = qubit_sites(n)
  else
    n = length(sites)
  end

  psi = initial_state(sites)

  gates = make_gates(sites)

  psi = tebd_gate_evolution(gates, psi; cutoff, maxdim)

  return psi
end
