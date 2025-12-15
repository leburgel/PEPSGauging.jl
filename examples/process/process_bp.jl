using TensorKit
using PEPSKit
using CairoMakie
using JLD2

include(joinpath(pwd(), "tools.jl"))
resdir = joinpath(pwd(), "results")

## Set run parameters

gauge = "bp"

D = 3
chi = 20
chi1 = 50
symmetrization = nothing # RotateReflect()
reuse_env = true
Jx = -1.0
Jy = 1.0
Jz = -1.0

mksz = 6

# load the data

fname = generate_heisenberg_filename(
    D, chi, chi1, gauge, symmetrization, Jx, Jy, Jz,
)
data = load(joinpath(resdir, fname))

n = length(data["es0"])
@show n

window = 1:n

## Plot energies

# real part: optimized versus check value
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"E")
scatterlines!(ax, window, real.(data["es0"][window]), label = "e0")
scatterlines!(ax, window, real.(data["es1"][window]), label = "e1")
axislegend(ax; position = :rt)
save(joinpath(resdir, "$(gauge)_e0_e1_nogauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

# imaginary part
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"|im(E)|", yscale = log10)
ydata = abs.(imag.(data["es0"][window]))
scatterlines!(ax, window, ydata)
ylims!(ax, minimum(ydata[ydata .!= 0.0]) / 2, maximum(ydata) * 2) # patch for exact zero entries
save(joinpath(resdir, "$(gauge)_e0_im_gauge_nogauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

## Plot gradient norms

# nogauge
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"||\vec{\nabla} f||", yscale = log10)
scatterlines!(ax, window, real.(data["ngs"][window]), markersize = mksz)
save(joinpath(resdir, "$(gauge)_ng_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

nothing
