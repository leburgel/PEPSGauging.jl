module BPAD

using TensorKit
using PEPSKit
using ChainRulesCore
using KrylovKit
using OptimKit
using Zygote

using PEPSKit: @autoopt, hook_pullback
using PEPSKit: CTMRGAlgorithm, GradMode, SymmetrizationStyle
using PEPSKit: PEPSTensor, PEPSWeight, unitcell
using PEPSKit: peps_normalize, real_inner
using PEPSKit: north_virtualspace, east_virtualspace, south_virtualspace, west_virtualspace
using PEPSKit: gauge_fix

# overloads
import PEPSKit: _rrule, update!

include("utils.jl")
include("piracy.jl")
include("fixedpoint.jl")
include("derivative.jl")
include("optimization.jl")

export gauge_preserving_fixedpoint

end
