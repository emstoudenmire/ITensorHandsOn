include("../tebd_gate_evolution.jl")

"""
  hamiltonian_evolution

Keyword arguments:
* `tau` (default: 0.05) - time step used. Smaller can be more accurate, but too small incurs 
                          a large amount of truncation error per unit time.
* `n` (default: 40) - number of sites
* cutoff - total "weight" of density matrix eigenvalues to truncate on each bond
* maxdim - maximum allowed bond dimension of tensor network after each step

"""
function hamiltonian_evolution(ttotal; tau=0.05, n=40, cutoff=default_cutoff(), maxdim=default_maxdim())
  sites = siteinds("S=1/2",n)
  psi = initial_state(sites)

  # TODO #1 adapt the code at the following link
  # https://docs.itensor.org/ITensorMPS/stable/tutorials/MPSTimeEvolution.html
  # to insert your own code below that makes quantum gates which are a 
  # "Trotterized" or time-split evolution operator for the 
  # Heisenberg Hamiltonian discussed on that page
  #
  gates = ITensor[]

  for t in 0.0:tau:ttotal
    #
    # Measurement code goes here.
    #
    println("t = $t")
    t≈ttotal && break
    psi = tebd_gate_evolution(gates, psi; cutoff, maxdim)
    psi = normalize(psi)
  end

  # TODO #2 after implementing the gates and Hamiltonian evolution,
  # use the `ITensorMPS.expect` function or `ITensorMPS.correlation_matrix` 
  # to compute physical properties of the state at each time step.
  #
  # Insert this code inside the loop above.
  #
  # For example, these could be the expected value of "Sz" on each site.
  # Try plotting these, adjusting the total time, to see what results you get.
  # What tends to happen for longer times?

  return
end
