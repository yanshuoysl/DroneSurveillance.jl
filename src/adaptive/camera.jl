# Camera types are loaded before the POMDP type is defined.
const DSObservation = SVector{3, Int64}
const OBS_CLASSES = (:ordinary, :survey, :outside, :obstacle)
const N_CELL_CLASSES = length(OBS_CLASSES)
const TERMINAL_OBSERVATION = DSObservation(0, 0, 0)

"""
    KnownCamera(; correct_probability=0.9)
    KnownCamera(confusion_matrix)

Known per-cell classification model. Matrix rows are true classes and columns
are reported classes, both ordered ordinary, survey, outside, obstacle. The three camera
readings are conditionally independent given the state. This is a simulation
model, with configurable probabilities rather than calibrated hardware data.
"""
struct KnownCamera
    confusion::SMatrix{4, 4, Float64, 16}

    function KnownCamera(confusion::AbstractMatrix{<:Real})
        size(confusion) == (4, 4) || throw(ArgumentError("camera matrix must be 4×4"))
        all(p -> isfinite(p) && 0 <= p <= 1, confusion) ||
            throw(ArgumentError("camera probabilities must be finite and between 0 and 1"))
        all(row -> isapprox(sum(row), 1.0; atol=1e-12, rtol=0), eachrow(confusion)) ||
            throw(ArgumentError("each camera matrix row must sum to 1"))
        return new(SMatrix{4, 4, Float64, 16}(confusion))
    end
end

function KnownCamera(; correct_probability::Real=0.9)
    isfinite(correct_probability) && 0 <= correct_probability <= 1 ||
        throw(ArgumentError("correct_probability must be between 0 and 1"))
    confusion = fill((1.0 - correct_probability)/(N_CELL_CLASSES - 1), N_CELL_CLASSES, N_CELL_CLASSES)
    for i in 1:N_CELL_CLASSES
        confusion[i, i] = correct_probability
    end
    return KnownCamera(confusion)
end

"""Return human-readable camera classes in front-left, center, right order."""
function observation_labels(o::DSObservation)
    o == TERMINAL_OBSERVATION && return (:terminal, :terminal, :terminal)
    return map(i -> OBS_CLASSES[i], Tuple(o))
end
