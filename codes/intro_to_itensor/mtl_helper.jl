using ITensors.NDTensors.Expose: Exposed
using Metal: MtlArray
using ITensors.NDTensors: NDTensors

function Base.print_array(io::IO, E::Exposed{<:MtlArray})
  return Base.print_array(io, expose(NDTensors.cpu(E)))
end