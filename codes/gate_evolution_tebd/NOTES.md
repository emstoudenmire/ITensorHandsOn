TEBD Gate Evolution
--------------------------------

This ITensor-based code activity asks the user to complete
key steps of a time-evolving block decimation (TEBD) code
for applying gates (two-site operators) to a quantum state,
with the state represented as an MPS tensor network.

Coding activities
--------------------------------

0. Read through the `tebd_time_evolution.jl` code.
   Find the section labeled "To Do" and fill in the missing
   code described there.

1. When you feel confident that your TEBD code is working,
   include the file `01_check_fidelity/check_fidelity.jl` 
   into a running Julia session and run the `check_fidelity(n)`
   function for small number of qubits `n` (n < 30) to check that
   your TEBD code is giving accurate results.

2. 


Tips for running the codes
--------------------------------

Open the Julia console (or "REPL") and load the codes like:
julia> include("tebd_gate_evolution.jl"); main()
