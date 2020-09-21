using StatsBase, LinearAlgebra
include("neuralNet.jl")
include("propagation.jl")
include("pulsesGandD.jl")

Eα_r = LinRange(0.0,1.25,1200)
δE = Eα_r[2] - Eα_r[1]
Eαr = h5read("Energy_V_alpha_beta.h5", "Energy_Levels");

data2 = h5open("dataSynthst10kfv2.h5", "r")
coeffsNoi = read(data2["coeffsNoi"])
xdata = reshape(coeffsNoi, (60,10*10000));

coeffsRefn = read(data2["coeffsRef"]);
bestTs_ref = read(data2["bestTs_ref"]);
ydata = coeffsRefn;

#duplicate to pair up with all references double pulse
function get_ydata(ref_ydata,i,j; nc= 40,sampref = 10)
    # i,j = 1,1, Guassian pulses
    outdata = zeros(Float64, nc, size(ref_ydata)[4]*sampref)
    ijdata = ref_ydata[:, i,j,:]
    increment = 1
    for i in 1:size(ydata)[4]
        outdata[:, increment:increment+sampref-1] = repeat(ijdata[:,i],1,sampref)
        increment += sampref
    end
    outdata
end
function splitdata(xdata, ydata)
    #μnoi = mean(xdata)
    #stdnoi = std(xdata)
    #μref = mean(ydata_1)
    #stdref = std(ydata_1)
    #xdata = (xdata .- μnoi)./stdnoi
    #ydata = (ydata .- μref)./stdref

    xtrain = xdata[:,1:70000]
    ytrain = ydata[:,1:70000]
    xval = xdata[:,70001:80000]
    yval = ydata[:,70001:80000]
    xtest = xdata[:,80001:end]
    ytest = ydata[:,80001:end]
    xtrain, ytrain, xval, yval, xtest, ytest #, [μnoi, stdnoi, μref, stdref]
end

i = 3
j = 3
ydataDij = get_ydata(ydata,i,j)
xtrain, ytrain, xval, yval, xtest, ytest = splitdata(xdata, ydataDij)
dict_models, losses = ensembleMachines(xtrain, ytrain, xval, yval;
                                                    nmodels = 10, nepoch = 100)
