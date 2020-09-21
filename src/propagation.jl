"""
discretePulse(pulse, gridPot ; rpulse = false)

A pulse is discretized given by a fixed grid-array of values from a general
reference pulse.
The output is an array with the _index positions_ of the corresponding values
from the grid-array. Also, is posible to obtain a _new pulse_ with the
corresponding values.
"""
function discretePulse(pulse, gridPot ; rpulse = false)
    ΔPulse = zeros(length(pulse))
    indxΔts = zeros(Int64,length(pulse))
    for (indx, fᵢ) in enumerate(pulse)
        gaIndx = argmin((fᵢ .- gridPot).^2)
        ΔPulse[indx] = gridPot[gaIndx]
        indxΔts[indx] = gaIndx
    end
    rpulse == true ? (indxΔts, ΔPulse) : indxΔts
end

"""
getΔtsPulse(tlims, pulse, gridPot)

Computes new Δts from a discretized pulse given the general reference pulse.
The output is a new set of _Δts_ and the corresponding _indices_ for the values
in the reference pulse.
"""
function getΔtsPulse(tlims, pulse, gridPot)
    indxΔts = discretePulse(pulse, gridPot)
    tᵢ = indxΔts[1]
    setΔtij = []
    indxi = 1
    uindxΔts = [tᵢ]
    for (index, tⱼ) in enumerate(indxΔts)
        if tⱼ != tᵢ
            δtij = (index -indxi)*dt
            push!(setΔtij, δtij)
            tᵢ = tⱼ
            push!(uindxΔts, tᵢ)
            indxi = index
        elseif index==length(indxΔts) && tⱼ == tᵢ
            δtij = (index - indxi + 1)*dt
            push!(setΔtij, δtij)
        end
    end
    setΔtij, uindxΔts
end

"""
diagLightMatter(filename, gridPot, Eα, Vαβ; dimBase=600)

Diagonalization of a given discretized range of values in the vector potential.
"""
function diagLightMatter(filename, gridPot, Eα, Vαβ; dimBase=600)
    sizeGridPot = size(gridPot)[1]
    filediag = h5open(filename * ".h5", "w")
    dset_psi = d_create(filediag,"psi", datatype(Complex{Float64}), dataspace(dimBase, dimBase, sizeGridPot))
    dset_invpsi = d_create(filediag,"invpsi", datatype(Complex{Float64}), dataspace(dimBase, dimBase, sizeGridPot))
    dset_energ = d_create(filediag,"energies", datatype(Complex{Float64}), dataspace(dimBase, sizeGridPot))
    p = Progress(sizeGridPot, dt=0.2, desc="diagonalizing...", color=:blue)
    for (i, aᵢ) in enumerate(gridPot)
        H̃αβ = Diagonal(Eα) .- aᵢ*Vαβ
        Ẽα, ψ̃α = eigen!(H̃αβ)
        dset_invpsi[:, :, i] = inv(ψ̃α)
        dset_psi[:, :, i] = ψ̃α
        dset_energ[:, i] = Ẽα
        szfile = filesize(filename * ".h5")
        ProgressMeter.next!(p; showvalues = [(:Filesize, szprintp(szfile))])
        IJulia.clear_output(true) # remove this if you run it in the terminal
    end
    flush(filediag)
    close(filediag)
end
function diagLightMatterS(gridPot, Eα, Vαβ; dimBase=600)
    sizeGridPot = size(gridPot)[1]
    dset_psi = zeros(Complex{Float64}, dimBase, dimBase, sizeGridPot)
    dset_invpsi = zeros(Complex{Float64},dimBase, dimBase, sizeGridPot)
    dset_energ = zeros(dimBase,sizeGridPot)
    #p = Progress(sizeGridPot, dt=0.2, desc="diagonalizing...", color=:blue)
    for i in 1:sizeGridPot
        H̃αβ = Diagonal(Eα) .- gridPot[i]*Vαβ
        Ẽα, ψ̃α = eigen!(H̃αβ)
        dset_invpsi[:, :, i] = inv(ψ̃α)
        dset_psi[:, :, i] = ψ̃α
        dset_energ[:, i] = real.(Ẽα)
       # ProgressMeter.next!(p)
    end
    dset_psi, dset_invpsi, dset_energ
end

"""
timeEvolution(Δtij, posVecPot, ncut = 600)

Time evolution for a given state ψt₀ (provided by default) is computed,
for varying times and strength's potentials. Before using this function
please run _getΔtsPulse_.
"""
function timeEvolution(Δtij, posVecPot, fψ̃α, finvψ̃α, fEα, ncut = 600)
    ψt₀ = zeros(Complex{Float64}, ncut)
    ψt₀[1] = 1
    for (index, pᵢ) in enumerate(posVecPot)
        sψ̃α = @view fψ̃α[:,:,pᵢ]
        sinvψ̃α = @view finvψ̃α[:,:,pᵢ]
        sEα = @view fEα[:,pᵢ]
        sΔt = @view Δtij[index]
        # time evolution
        ψtᵢ = sψ̃α * (exp.(-1im*real(sEα) .* sΔt) .* (sinvψ̃α * ψt₀))
        # be careful with the order in operations and the change from * to .*
        ψt₀ .= ψtᵢ
    end
    ψt₀
end

"""
timeEvolutionCN(pulse_ex, Vαβ, Eα, dt, ncut = 600)

Applying the Crank-nicolson method th time evolution for a given state ψt₀
(provided by default) is computed,for varying times and strength's potentials.
"""
function timeEvolutionCN(pulse, Vαβ, Eα, dt, ncut = 600)
    ψt₀ = zeros(Complex{Float64}, ncut)
    ψt₀[1] = 1
    p = Progress(size(pulse)[1], dt=0.01, desc="propagating...", color=:blue)
    for aᵢ in pulse
        H̃αβ = Diagonal(Eα) .- aᵢ*Vαβ
        Uf = -1im .* dt .* H̃αβ./2 + I
        Ub = 1im .* dt .* H̃αβ./2 + I
        #evolution
        y = Uf * ψt₀
        ψt₀ =  Ub\y             # solves Ub*x = y, new ψt0
        ProgressMeter.next!(p)
        #sleep(0.05)
        IJulia.clear_output(true)
    end
    ψt₀
end

"""
gaussKernel(Eα_r, Eαi, ΔEw)

Gaussian kernel. Cented at Eαi and width ΔEw.
"""
gaussKernel(Eα_r, Eαi, ΔEw) = exp.(-(Eα_r .- Eαi).^2 ./ΔEw.^2)/(ΔEw*sqrt(π))


"""
getSpectrum(coeffs, ΔEw, Eα, Eα_r)

Computes the power spectrum from the photon-electron interaction. With a gaussian kernel.
Where _ΔEw_ is the width for the gaussian kernel, _Eα_ is the spectrum from the interaction region and
_coeffs_ are the coefficients after the evolution of an initial state ψt₀.
"""
function getSpectrum(coeffs, ΔEw, Eα, Eα_r)
    nf = length(Eα)
    spectrum = zeros(nf)
    square_coefs = abs.(coeffs).^2
    for i in 1:nf
        spectrum += square_coefs[i] .* gaussKernel(Eα_r, Eα[i], ΔEw)
    end
    spectrum
end

function getSpectrumCs(square_coefs, ΔEw, Eα, Eα_r)
    nf = length(Eα)
    spectrum = zeros(length(Eα_r))
    for i in 1:nf
        spectrum += square_coefs[i] .* gaussKernel(Eα_r, Eα[i], ΔEw)
    end
    spectrum
end

χ0(xs, Ωs) = (Ωs/π)^(1/4) .* exp.(- Ωs*xs.^2.0./2)
χ1(xs, Ωs) = (4Ωs^3/π)^(1/4) .* xs .* exp.(- Ωs*xs.^2.0./2)
function χn(x; m = 2, t0 = 3, ω₀ = 0.57, Eα0 = -0.9)
    nr = m - 1
    T0 = 41.34*t0
    Ωs = T0^2/(4*log(2))
    x₀ = Eα0 .+ 2ω₀
    xs = (x .- x₀)
    a, b = χ1(xs, Ωs), χ0(xs, Ωs)
    if m<0
        return println("m must be greater than zero")
    elseif m==0
        return b
    elseif  m==1
        return a
    else
        for n in 1:nr
            a, b = √(2Ωs/(n+1)) .* xs .* a .-  √(n/(n+1)) .* b, a
        end
    end
    return a
end

"""
posnegE(Eαn)

Returns: Int64, bound position of positive values

"""
function posnegE(Eαn; valor = 0)
    nbound = 0
    for i in 1:length(Eαn)
        if Eαn[i] >valor
            nbound = i - 1
            break
        end
    end
    nbound
end

"""
randEαVαβ(Eα, Vαβ; γ = 0.88, upperlimit = 1)

Returns: Synthetic elements for new Hamiltonians (synthetic hamiltonians).
"""
function randEαVαβ(Eα, Vαβ; γ = 0.88, lowerlimit = -1, upperlimit = 1)
    Eαn = copy(Eα)
    Vαβn = copy(Vαβ)
    posEnerg = posnegE(Eαn)
    χ2, χ3, χ4 = 2*rand(3) .+ lowerlimit
    χ1 = (upperlimit - lowerlimit) * rand() .+ lowerlimit

    Eαn[2:posEnerg] .= 3^(χ1 - γ) * Eαn[2:posEnerg]
    Vαβn[1,1:posEnerg] .= 3^(χ2)* Vαβn[1,1:posEnerg]
    Vαβn[1:posEnerg,1] .= 3^(χ2) * Vαβn[1:posEnerg,1]
    Vαβn[1:posEnerg,posEnerg+1:end] .= 3^(χ3)* Vαβn[1:posEnerg,posEnerg+1:end]
    Vαβn[posEnerg+1:end,1:posEnerg] .= 3^(χ3)* Vαβn[posEnerg+1:end,1:posEnerg]
    Vαβn[posEnerg+1:end,posEnerg+1:end] .= 3^(χ4)* Vαβn[posEnerg+1:end,posEnerg+1:end]
    Eαn, Vαβn
end

function findCns(Eα_r, spectra, δE, Ntot, tt0, omega0; Eg = Eαr[1])
    Cᵢs = zeros(Ntot)
    for nn in 0:Ntot-1
        Cᵢs[nn+1] = sum(sqrt.(spectra) .* χn(Eα_r; m = nn, t0 = tt0, ω₀ = omega0, Eα0 = Eg))*δE
    end
    Cᵢs
end

function fitSpectro(Eα_r, coefref, tt0, omega0; Eg = Eαr[1])
    ncoefi = length(coefref)
    spectFitref = zeros(length(Eα_r))
    for i in 1:ncoefi
        spectFitref += coefref[i] * χn(Eα_r; m =i-1, t0=tt0,ω₀ =omega0,
            Eα0= Eg)
    end
    abs.(spectFitref).^2
end

function noisyStates(fψ̃α, finvψ̃α, fEα; npulses=1000, gref=false, tt0 = 3,
    τ0 = 0.5, omegac = 0.77, fflu = 1.0, vseed=1021)
    fstates = zeros(600, npulses)
    Random.seed!(vseed)
    for pulso in 1:npulses
        tlims_ex, pulse_ex =  FELpulse(t, Nt; T = tt0, τ = τ0, ω₀ = omegac,
        gauss_ref = gref, fluence = fflu)
        Δtij, posVecPot = getΔtsPulse(tlims_ex, pulse_ex, gridPot)
        ψt₀ = timeEvolution(Δtij, posVecPot, fψ̃α, finvψ̃α, fEα)
        fstates[:,pulso] = abs.(ψt₀).^2
    end
    fstates
end
function stateRefs(fψ̃α, finvψ̃α, fEα; tt0 = 3, omegac = 0.77, gaFlux = 1.0)
    tlims_ex, pulse_ex = pulseGauss(t, tt0, 1e10, omegac, gaFlux)
    Δtij, posVecPot = getΔtsPulse(tlims_ex, pulse_ex, gridPot)
    ψt₀ = timeEvolution(Δtij, posVecPot, fψ̃α, finvψ̃α, fEα)
    abs.(ψt₀).^2
end

function stateRefsDoubles(fψ̃α, finvψ̃α, fEα; tt0 = 3, omegac = 0.77,gaFlux = 1.0)
    tlims_ex, pulse_ex = pulsedouble(t, tt0,1e10, omegac, gaFlux)
    Δtij, posVecPot = getΔtsPulse(tlims_ex, pulse_ex, gridPot)
    ψt₀ = timeEvolution(Δtij, posVecPot, fψ̃α, finvψ̃α, fEα)
    abs.(ψt₀).^2
end

function stateRefsDoubles(fψ̃α, finvψ̃α, fEα, separation; tt0 = 3, omegac = 0.77,
    gaFlux = 1.0)
    tlims_ex, pulse_ex = pulsesSep(t,tt0,separation,omegac,gaFlux)
    Δtij, posVecPot = getΔtsPulse(tlims_ex, pulse_ex, gridPot)
    ψt₀ = timeEvolution(Δtij, posVecPot, fψ̃α, finvψ̃α, fEα)
    abs.(ψt₀).^2
end

function fbeststar(Eα_r, refspectra, specFitted,besT₀, Cns, difFit,
     δE, ncoef, cfreq)
    for i in 0.01:0.01:4
        Cnsx = findCns(Eα_r, refspectra, δE, ncoef, i, cfreq)
        specFittedx = fitSpectro(Eα_r, Cnsx, i, cfreq)
        difFitx = sum(abs.(refspectra .- specFittedx))*δE
        if difFitx <= difFit
            besT₀ = i
            Cns = Cnsx
            difFit = difFitx
            specFitted = specFittedx
        end
    end
    besT₀, Cns, difFit, specFitted
end
