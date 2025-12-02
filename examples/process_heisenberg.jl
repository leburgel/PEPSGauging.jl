using CairoMakie
using JLD2

include(joinpath(@__DIR__, "tools.jl"))
resdir = joinpath(@__DIR__, "results")
## Set run parameters

D = 3
chi = 20
chi1 = 50
symmetrization = nothing # RotateReflect()
reuse_env = true
Jx = -1.0
Jy = 1.0
Jz = -1.0

# load the data

nogauge_name = generate_heisenberg_filename(
    D, chi, chi1, false, symmetrization, Jx, Jy, Jz,
)
nogauge_data = load(joinpath(resdir, nogauge_name))

gauge_name = generate_heisenberg_filename(
    D, chi, chi1, true, symmetrization, Jx, Jy, Jz,
)
gauge_data = load(joinpath(resdir, gauge_name))

n = length(nogauge_data["es0"])

## Plot energies: optimized value and check value

# nogauge
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"E")
scatterlines!(ax, 1:n, real.(nogauge_data["es0"]), label = "e0")
scatterlines!(ax, 1:n, real.(nogauge_data["es1"]), label = "e1")
axislegend(ax; position = :rt)
save(joinpath(resdir, "e0_e1_nogauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

# gauge
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"E")
scatterlines!(ax, 1:n, real.(gauge_data["es0"]), label = "e0")
scatterlines!(ax, 1:n, real.(gauge_data["es1"]), label = "e1")
axislegend(ax; position = :rt)
save(joinpath(resdir, "e0_e1_gauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

## Plot energies: nogauge versus gauge

# real part
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"E")
scatterlines!(ax, 1:n, real.(nogauge_data["es0"]), label = "nogauge")
scatterlines!(ax, 1:n, real.(gauge_data["es0"]), label = "gauge")
axislegend(ax; position = :rt)
save(joinpath(resdir, "e0_gauge_nogauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

# imaginary part
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"|im(E)|", yscale = log10)
scatterlines!(ax, 1:n, abs.(imag.(nogauge_data["es0"])), label = "nogauge")
scatterlines!(ax, 1:n, abs.(imag.(gauge_data["es0"])), label = "gauge")
axislegend(ax; position = :rb)
save(joinpath(resdir, "e0_im_gauge_nogauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)

## Plot gradient norms: nogauge versus gauge

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "iteration", ylabel = L"||\vec{\nabla} f||", yscale = log10)
scatterlines!(ax, 1:n, nogauge_data["ngs"], label = "nogauge")
scatterlines!(ax, 1:n, gauge_data["ngs"], label = "gauge")
axislegend(ax; position = :rt)
save(joinpath(resdir, "ng_gauge_nogauge_D_$(D)_chi_$(chi)_chi1_$(chi1).png"), fig)
