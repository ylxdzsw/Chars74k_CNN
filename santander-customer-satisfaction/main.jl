using DataFrames
using JLD

include("../../krife.jl/src/krife.jl")
using krife

# using PyCall

# @pyimport sklearn.ensemble as skes
# @pyimport sklearn.cross_validation as skcv

"===reading===" |> println
@time begin
    @load "processed.jld" trainX trainY testX
end

"===reformating===" |> println
@time begin
    trainX = df_all[1:length(trainY),:]      |> DataArray
    testX  = df_all[length(trainY)+1:end, :] |> DataArray
    df_all = nothing
end

"===xgboosting===" |> println
@time let
    writelibsvm("train.libsvm", trainX, trainY)
    writelibsvm("test.libsvm", testX)

    run(`bash -c "xgboost train.xgboost.conf"`)
    run(`bash -c "xgboost test.xgboost.conf"`)

    result = readcsv("pred.txt")
    submit = readtable("sample_submission.csv")
    submit[:TARGET] = result[:,1]
    writetable("submit.csv", submit)
end

# "===extra tree===" |> println
# @time begin
#     model = skes.ExtraTreesClassifier(n_estimators = 100)

#     a = skcv.cross_val_score(model, trainX, trainY, scoring="roc_auc", cv=3)

#     model[:fit](trainX, trainY)
#     testY = model[:predict_proba](testX)

#     submit = readtable("sample_submission.csv")
#     submit[:TARGET] = testY[:,2]
#     writetable("submission.csv", submit)
# end

