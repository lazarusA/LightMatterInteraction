using Flux, HDF5, ProgressMeter
using MLDataUtils, BSON
using Statistics, Random
using Base.Iterators: partition
#regression neural net
function regressionNN(dimIn, dimOut, neuronas)
    Chain(
        Dense(dimIn, neuronas, Flux.relu),
        Dense(neuronas, neuronas, Flux.relu),
        Dense(neuronas, neuronas, Flux.relu),
        Dense(neuronas, neuronas, Flux.relu),
        Dense(neuronas, neuronas, Flux.relu),
        #Dense(neuronas, neuronas, Flux.relu),
        Dense(neuronas, neuronas, Flux.relu),
        Dense(neuronas, dimOut)) # |> gpu
end
function machine(xdata, ydata, xval, yval, seed;
        neurons=40, nepochs=100, bs=512, best_val = 5.0,
        last_improvement = 0,patience=5, gpu = false)
    #saving best model
    dict_models = Dict()
    dict_models["model_bm$(seed)"] = NaN
    #optimizer and loss function
    Random.seed!(seed)
    model = regressionNN(size(xdata)[1], size(ydata)[1], neurons)
    # if training on gpu's is an option
    model = gpu == true ? model |> gpu : model
    xdata = gpu == true ? xdata |> gpu : xdata
    ydata = gpu == true ? ydata |> gpu : ydata
    
    opt = ADAM()
    loss(x,y) = Flux.mse(model(x), y)
    #nepochs = 100
    lossVal = zeros(nepochs)
    #mini-batch
    #batch_size 512 is ok too, similar performance at test accuracy
    mb_idxs = partition(1:size(xdata)[2], bs)
    #xs, ys = shuffleobs((Float32.(xdatacoef), ycdata))
    p = Progress(nepochs, dt=1, barglyphs=BarGlyphs("[=> ]"), color = :yellow)
    for epoch_indx in 1:nepochs
        xs, ys = shuffleobs((Float32.(xdata), Float32.(ydata)))
        train_set = [(xs[:,p], ys[:,p]) for p in mb_idxs]
        Flux.train!(loss, params(model), train_set, opt)
        
        validation_loss = loss(xval, yval)
        lossVal[epoch_indx] = validation_loss
        # If this is the best val_loss we've seen so far, save the model out
        if validation_loss <= best_val
            #@info(" -> New best val_loss! Saving model out to model_bm$(seed).bson")
            last_improvement = epoch_indx
            dict_models["model_bm$(seed)"] = model # model_bestmodel 
            best_val = validation_loss
        end
        if epoch_indx - last_improvement >= patience
            #@info(" -> Early-exiting iteration $(r) and epoch $(epoch_indx): no more patience")
            break
        end
        next!(p; showvalues = [(:epoch,epoch_indx), (:Model_seed, seed)])
        #ProgressMeter.ijulia_behavior(:append)
        ProgressMeter.ijulia_behavior(:clear)
    end
    model = dict_models["model_bm$(seed)"]
    #BSON.@save "models_bm$(seed).bson" dict_models lossVal
    model, lossVal
end

function ensembleMachines(xtrain, ytrain, xval, yval; 
        nmodels = 10, nepoch = 100, neurons=40, bs = 512, best_val = 5.0,
        last_improvement = 0, patience=10, gpu = false)
    
    losses = zeros(nepoch, nmodels)
    machines = []
    dict_models = Dict()
    #pro = Progress(nmodels, dt=0.5, barglyphs=BarGlyphs("[=> ]"), color = :black)
    for seed in 1:nmodels
        modelo1, lossVal = machine(xtrain, ytrain, xval, yval,seed;
            neurons= neurons, nepochs= nepoch, bs=bs, best_val = best_val,
            last_improvement = last_improvement, patience=patience, gpu = gpu)
        
        losses[:,seed] = lossVal
        dict_models["model_bm$(seed)"] = modelo1
        #next!(pro; showvalues = [(:Model, seed)])
        #ProgressMeter.ijulia_behavior(:append)
        #ProgressMeter.ijulia_behavior(:clear)
    end
    #BSON.@save "models_bm$(nmodels).bson" dict_models losses
    dict_models, losses
end

function splitdata(xdata, ydata)
    #μnoi = mean(xdata)
    #stdnoi = std(xdata)
    #μref = mean(ydata_1)
    #stdref = std(ydata_1)
    #xdata = (xdata .- μnoi)./stdnoi
    #ydata = (ydata .- μref)./stdref

    xtrain = xdata[:,1:32000]
    ytrain = ydata[:,1:32000]
    xval = xdata[:,32001:36000]
    yval = ydata[:,32001:36000]
    xtest = xdata[:,36001:end]
    ytest = ydata[:,36001:end]
    xtrain, ytrain, xval, yval, xtest, ytest #, [μnoi, stdnoi, μref, stdref]
end
