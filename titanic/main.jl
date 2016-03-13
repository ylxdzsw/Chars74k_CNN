using PyCall
using DataFrames

@pyimport sklearn.ensemble as skes
@pyimport sklearn.cross_validation as skcv

df_train = readtable("train.csv")
df_test = readtable("test.csv")

featureList = [:Pclass, :Age, :SibSp,
               :Parch, :Fare]

imputena!(x) = begin # meanAge should include both train and test
    meanAge = x[:Age] |> dropna |> mean
    x[:Age][x[:Age].na] = meanAge
    meanFare = x[:Fare] |> dropna |> mean
    x[:Fare][x[:Fare].na] = meanFare
    x
end

trainX = df_train[featureList] |> imputena! |> Array{Float32}
trainY = df_train[:Survived] |> Array{Int}
testX  = df_test[featureList] |> imputena! |> Array{Float32}

model = skes.RandomForestClassifier(n_estimators = 100)

a = skcv.cross_val_score(model, trainX, trainY, cv=3)
