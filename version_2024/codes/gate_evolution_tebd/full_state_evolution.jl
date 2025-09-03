using ITensors
using Random

include("util.jl")
include("make_gates.jl")

function full_state_evolution(gates, psi)
  for g in gates
    psi = noprime(g*psi)
  end
  return psi
end

function main(; n=10, seed=1)
  Random.seed!(seed)

  if isnothing(sites)
    sites = qubit_sites(n)
  else
    n = length(sites)
  end

  psi = prod(initial_state(sites))

  gates = make_gates(sites)

  psi = full_state_evolution(gates,psi)

  return psi
end
