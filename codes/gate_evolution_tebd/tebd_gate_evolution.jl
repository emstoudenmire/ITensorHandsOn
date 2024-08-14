using ITensors
using ITensorMPS
using Random

include("util.jl")
include("make_gates.jl")

function apply_gate(g, psi; cutoff, maxdim)
  j = findsite(psi,inds(g))  # the gate acts on sites j and j+1
  psi = orthogonalize(psi,j) # this is a technical step to 
                             # ensure truncation accuracy

  # Task TODO:
  # Implement the "TEBD" gate application method here, 
  # using ITensor contraction A*B, the "noprime" function,
  # and the ITensor svd function.
  # Tips:
  # - You can obtain the j and j+1 MPS tensors like psi[j] and psi[j+1].
  # - The uniqueinds and commoninds functions can be helpful to obtain collections of
  #   indices to pass to the svd function.
  # - Pass cutoff and maxdim to the svd to control the resulting "bond dimension"
  # - Don't forget to normalize the wavefunction! You can use norm(T) to 
  #   get the norm (sqrt of sum of squared elements) of an ITensor.

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
