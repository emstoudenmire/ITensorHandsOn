include("full_state_evolution.jl")
include("tebd_gate_evolution.jl")

function check_fidelity(; n=10)
  Random.seed!(1)

  sites = qubit_sites(n)
  gates = make_gates(sites)

  psi0 = initial_state(sites)

  psi_x = full_state_evolution(gates,prod(psi0))
  psi_m = tebd_gate_evolution(gates,psi0)

  overlap = scalar(psi_x*prod(psi_m))

  println("overlap = ",overlap)
end
