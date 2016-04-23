using DataFrames
include("../../krife.jl/src/krife.jl")
using krife

# using PyCall

# @pyimport sklearn.ensemble as skes
# @pyimport sklearn.cross_validation as skcv

"===reading===" |> println
gc()
@time begin
    df_train = readtable("train.csv")
    df_test  = readtable("test.csv")
end

"===basic formating===" |> println
gc()
@time begin
    features = [:var15,:saldo_var30,:std,:num_var22_ult3,:imp_op_var39_ult1,:num_var45_hace3,:saldo_medio_var5_hace2,:var3,:saldo_medio_var8_ult3,:ind_var41_0]
    trainX = df_train[features] |> DataArray
    testX  = df_test[features]  |> DataArray
    trainY = df_train[:TARGET]
end

"===xgboost===" |> println
gc()
@time begin
    writelibsvm("train.libsvm", trainX, trainY)
    writelibsvm("test.libsvm", testX)

    run(`xgboost train.xgboost.conf`)
    run(`xgboost test.xgboost.conf`)

    result = readcsv("pred.txt")
    submit = readtable("sample_submission.csv")
    submit[:PredictedProb] = result[:,1]
    writetable("submit.csv", submit)
end

# "===extra tree===" |> println
# gc()
# @time begin
#     model = skes.ExtraTreesClassifier(n_estimators = 100)

#     a = skcv.cross_val_score(model, trainX, trainY, scoring="roc_auc", cv=3)

#     model[:fit](trainX, trainY)
#     testY = model[:predict_proba](testX)

#     submit = readtable("sample_submission.csv")
#     submit[:TARGET] = testY[:,2]
#     writetable("submission.csv", submit)
# end

