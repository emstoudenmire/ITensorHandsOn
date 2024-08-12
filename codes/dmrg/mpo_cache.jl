
struct MPOCache
  lpos::Int
  rpos::Int
  cache::Dict{Pair{Int,Int},ITensor}
  LH::ITensor
  RH::ITensor
end

function MPOCache(H::MPO) 
  N = length(H)
  cache = Dict{Pair{Int,Int},ITensor}()
  cache[0=>1] = ITensor(1.0)
  cache[N+1=>N] = ITensor(1.0)
  return MPOCache(1,N,cache,ITensor(),ITensor())
end

function position(C::MPOCache, H::MPO, psi::MPS, bond)
  newlpos = min(bond...)
  newrpos = max(bond...)
  lpos = C.lpos
  rpos = C.rpos
  cache = copy(C.cache)
  lpos = min(lpos,newlpos)
  while lpos < newlpos
    cache[lpos=>lpos+1] = ((cache[lpos-1 => lpos]*psi[lpos])*H[lpos])*dag(prime(psi[lpos]))
    lpos += 1
  end
  @assert lpos==newlpos
  rpos = max(rpos,newrpos)
  while rpos > newrpos
    cache[rpos=>rpos-1] = ((cache[rpos+1 => rpos]*psi[rpos])*H[rpos])*dag(prime(psi[rpos]))
    rpos -= 1
  end
  @assert rpos==newrpos
  return MPOCache(lpos,rpos,cache,H[lpos],H[rpos])
end

left_environment(C::MPOCache) = C.cache[C.lpos-1=>C.lpos]
right_environment(C::MPOCache) = C.cache[C.rpos+1=>C.rpos]

left_H(C::MPOCache) = C.LH
right_H(C::MPOCache) = C.RH
