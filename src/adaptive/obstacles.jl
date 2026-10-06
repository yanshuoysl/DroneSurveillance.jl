"""
    random_obstacles(size; count=2, exclude=DSPos[], rng=Random.default_rng())

Sample distinct obstacle cells without replacement, excluding reserved cells.
"""
function random_obstacles(size; count::Integer=2, exclude=DSPos[], rng=Random.default_rng())
    all(>(0), size) || throw(ArgumentError("grid dimensions must be positive"))
    candidates = [DSPos(xy[1], xy[2]) for xy in vec(collect(CartesianIndices(size)))
                  if DSPos(xy[1], xy[2]) ∉ exclude]
    0 <= count <= length(candidates) || throw(ArgumentError("not enough free cells for $count obstacles"))
    return shuffle(rng, candidates)[1:count]
end
