using DataFrames
include("../../krife.jl/src/krife.jl")
using krife

# using PyCall

# @pyimport sklearn.ensemble as skes
# @pyimport sklearn.cross_validation as skcv

"===reading===" |> println
@time begin
    trainX, trainY, testX = let
        df_train = readtable("train.processed.csv")
        df_test  = readtable("test.processed.csv")
        trainY = df_train[:TARGET]
        delete!(df_train, :TARGET)
        trainX = df_train |> DataArray
        testX  = df_test  |> DataArray
        trainX, trainY, testX
    end
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

