using ITensors, ITensorMPS

include("../tebd_gate_evolution.jl")

function compute_sample(psi::MPS)
  psi = orthogonalize(psi,1)
  n = length(psi)

  sample = zeros(Int,n)

  A = psi[1]
  for j=1:n
    s = siteind(psi,j)
    d = dim(s)

    rho = prime(A,s) * dag(A)

    r = rand()
    sampled_value = 1

    #
    # TODO: Your task...
    # Implement code which does the following
    # - taking the diagonals `real(rho[i,i])`
    #   of the reduced density matrix
    # - taking the random value `r` computed for you
    #   above
    # - work out the correct sampled_value (∈ [0,d])
    #   to save into the `sample` array below
    #
    # Correct means such that the user gets a random
    # sample of the state, distributed according to 
    # the "Born rule" of quantum probability.
    #

    # your code goes here...

    #############################

    sample[j] = sampled_value
    if j < n
      A = psi[j + 1] * (A*onehot(dag(s)=>val))
      A /= norm(A)
    end
  end
  return sample
end

function test_sampling(n; nsamples=10, cutoff=default_cutoff(), maxdim=default_maxdim())
  sites = qubit_sites(n)
  psi = initial_state(sites)
  gates = make_random_gates(sites)
  psi = tebd_gate_evolution(gates, psi; cutoff, maxdim)

  for s=1:nsamples
    sample = compute_sample(psi)
    println("Sample #$s: ",sample)
  end
end
