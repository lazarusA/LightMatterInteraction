# Global parameters
Nt = 2^17
tmax = 3000
tmin = -3000
dt = (tmax - tmin)/Nt
t = collect(tmin:dt:tmax)[2:end]
α, linf, rsup, δx = 0.5, -500, 500,0.1
xd = linf:δx:rsup

dω= 2π/(Nt*dt)
ω = collect(-Nt/2:1:Nt/2)*dω
ω = ω[1:end-1];

ΔEw = 0.2/ 27.211396
Eα_r = LinRange(0.0,1,600)
δE = Eα_r[2] - Eα_r[1]
ncoef = 150
# Main functions


"""
tFELlimits(FELpulse, Nt, cut = 1e-4)

Computes a cut off of oscillations from the left and the right in the pulse.
Returns the left and right limits.

"""
function tFELlimits(FELpulse, Nt, cut = 1e-4)
    n_tmin = 0
    for i in 1:Nt
        if abs.(FELpulse[i] >= cut)
            n_tmin = i
            break
        end
    end
    n_tmax = 0
    for i in 1:Nt
        if abs.(FELpulse[Nt - i + 1] >= cut)
            n_tmax = Nt - i + 1
            break
        end
    end
    n_tmin, n_tmax
end

"""
G(t, T)

Pulse Envelope. Where T is the intensity FWHM time duration.
"""
G(t, T) = exp.(-2 .*log(2) .*(t ./T ).^2)

"""
g(t, τ)

Pulse. Where τ is the pulse duration (Pulse HW1/e).
In the PCM context is called coherence time.
"""
g(t, τ ) = exp.(-(t./ τ).^2)

"""
FELpulse(t, Nt; T = 3, τ = 0.5, ω₀ = 0.77, intensity = 1.0, fluence = 1,
        gauss_ref = false, asymm = true)

Generates pulses, via the Partial Coherence method (PCM):

\\begin{equation}
f\\_{l}(t) = N\\_{l} G(t) \\mathcal{F}^{-1}\\left[ e^{i\\phi\\_{l}(\\omega)}
\\mathcal{F}[g(t)\\cos(\\omega\\_{0}t)] \\right]
\\end{equation}

with

\\begin{equation}
G(t) = e^{-2\\ln 2\\left(\\frac{t}{T}\\right)^2},\\quad\\mbox{and}\\quad
g(t) = e^{-\\left(\\frac{t}{\\tau}\\right)^2},
\\end{equation}

where \$\\omega_{0}\$ is the carrier frequency and \$\\mathcal{F}\$ \$\\left(\\mathcal{F}^{-1}\\right)\$
the Fourier (inverse Fourier) transform. Also, \$\\tau\$ is the coherence time and \$T\$ the pulse duration.
\$ N\\_{l} \$ is a constant to normalize all pulses to unit pulse energy(default)
otherwise a different intensity(fluence) can be specified.

# Examples
```
julia> Nt = 2^17
julia> tmax = 3000
julia> tmin = -3000
julia> dt = (tmax - tmin)/Nt
julia> t = collect(tmin:dt:tmax)[2:end]
julia> tlims, pulse = FELpulse(t, Nt;) #gaussian pulse

```
"""
function FELpulse(t, Nt; T = 3, τ = 0.5, ω₀ = 0.77, intensity = 1.0,
    fluence = 1.0, gauss_ref = false, asymm = true)

    Tf = 41.34*T
    τf = 41.34*τ
    ϕp = gauss_ref == true ? zeros(Nt) : 2 .*π .*rand(Nt) .- π
    if asymm == true
        ϕp[2:Int64(Nt/2)] .= - reverse(ϕp[Int64(Nt/2 + 2):end]) # antisymmetric
    end
    flr = real(G(t, Tf) .* ifft(ifftshift(exp.(1im*ϕp).*fftshift(fft(g(t, τf) .* cos.(ω₀ .* t))))))
    pulse = flr  .* sqrt(intensity/(3.51ω₀^2))
    nnorm = sum(flr .^2) .* dt
    pulse = gauss_ref == true ? pulse : sqrt(fluence/nnorm) .* flr
    tmin, tmax = tFELlimits(pulse, Nt)
    t[tmin:tmax], pulse[tmin:tmax]
end

function pulsedouble(t, T, τ, ωs, gaFlux)
    Tf = 41.34*T
    τf = 41.34*τ
    dawsonVals = [dawson(i/τf) for i in t]
    flr = (2/√π)*exp.(-2*log(2)*(t/Tf).^2).* dawsonVals .* sin.(ωs .* t)
    nnorm = sum(flr .^2) .* dt
    pulse = sqrt(gaFlux/nnorm) .* flr
    tmin, tmax = tFELlimits(pulse, Nt)
    t[tmin:tmax], pulse[tmin:tmax]
end

function pulseGauss(t, T, τ, ω₀, intensity)
    Tf = 41.34*T
    τf = 41.34*τ
    flr = exp.(-2*log(2)*(t/Tf).^2 - (t/τf).^2) .* cos.(ω₀ .* t)
    pulse = flr  .* sqrt(intensity/(3.51ω₀^2))
    tmin, tmax = tFELlimits(pulse, Nt)
    t[tmin:tmax], pulse[tmin:tmax]
end

function pulsesSep(t,T1, τ1, ω₀, gaFlux)
    τ = 41.34*τ1
    T = 41.34*T1
    flr = (exp.(-2log(2).*(t .+ τ/2).^2/T^2) .+ exp.(-2log(2).*(t .- τ/2).^2/T^2)).*cos.(ω₀*t)
    nnorm = sum(flr .^2) .* dt
    pulse = sqrt(gaFlux/nnorm) .* flr
    tmin, tmax = tFELlimits(pulse, Nt)
    t[tmin:tmax], pulse[tmin:tmax]
end

function pulsesSep(t,T1, τ1, ω₀, frac, gaFlux)
    τ = 41.34*τ1
    T = 41.34*T1
    flr = (frac*exp.(-2log(2).*(t .+ τ/2).^2/T^2) .+ (1-frac)*exp.(-2log(2).*(t .- τ/2).^2/T^2)).*cos.(ω₀*t)
    nnorm = sum(flr .^2) .* dt
    pulse = sqrt(gaFlux/nnorm) .* flr
    tmin, tmax = tFELlimits(pulse, Nt)
    t[tmin:tmax], pulse[tmin:tmax]
end
