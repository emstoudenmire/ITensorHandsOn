
using OMEinsumContractionOrders 
using Graphs

function random_regular_eincode(n, k; optimize=nothing)
  g = Graphs.random_regular_graph(n, k)
  ixs = [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]
  vs = [[i] for i in Graphs.vertices(g)]
  return OMEinsumContractionOrders.EinCode([ixs..., vs...], Int[])
end

let
  code = random_regular_eincode(10, 3);

  optcode_tree = optimize_code(code, uniformsize(code, 2),
    TreeSA(sc_target=28, βs=0.1:0.1:10, ntrials=2, niters=100, sc_weight=3.0));

  C = contraction_complexity(code, uniformsize(code, 2))

  @show code
  @show optcode_tree
  @show C
  return
end
