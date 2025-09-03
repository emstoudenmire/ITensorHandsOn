include("../tebd_gate_evolution.jl")
include("../utilities/initial_state.jl")
include("../utilities/make_random_gates.jl")

"""
  full_state_evolution(gates, psi)

Given an array of gates as ITensors and a state tensor psi,
apply each gate to psi and return psi.
"""
function full_state_evolution(gates, psi)
  for g in gates
    psi = noprime(g*psi)
  end
  return psi
end

"""
  check_fidelity(; n, seed)

Given a small number of qubits `n` and an optional random seed, apply
a set of random quantum gates to a state in MPS form using the TEBD
algorithm, then apply the same set of gates to the same initial state
using an exponential-cost "full state" algorithm. 

Compute the  overlap between these to check the quality of 
the TEBD algorithm.
"""
function check_fidelity(n; seed=1)
  Random.seed!(seed)

  sites = qubit_sites(n)
  gates = make_random_gates(sites)

  psi0 = initial_state(sites)

  psi_mps = tebd_gate_evolution(gates,psi0)
  psi_full = full_state_evolution(gates,prod(psi0))

  overlap = scalar(psi_full*prod(psi_mps))

  println("overlap = ",overlap)
end
