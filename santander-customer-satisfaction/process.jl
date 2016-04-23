using DataFrames
using PyCall
using JLD

@pyimport sklearn.ensemble as skes
@pyimport sklearn.feature_selection as skfs

"===reading===" |> println
@time begin
    df_all, trainY = let
        df_train = readtable("train.csv")
        df_test  = readtable("test.csv")
        features = setdiff(names(df_test), [:ID])
        df_all   = vcat(df_train[features], df_test[features])
        trainY   = df_train[:TARGET]
        df_all, trainY
    end
end

"===droping constant columns===" |> println
@time let
    const_col = map(names(df_all)) do x
        std(df_all[x]) == 0
    end
    @show sum(const_col)
    df_all = df_all[!const_col]
end

"===droping duplicate columns===" |> println
@time let
    duplicate_col = map(1:length(df_all)) do x
        any(1:x-1) do y
            abs(cor(df_all[:,x], df_all[:,y])) > 0.98
        end
    end
    @show sum(duplicate_col)
    df_all = df_all[!duplicate_col]
end

"===inserting features===" |> println
@time let
    df_all[:num_zeros] = Int[sum(df_all[i,:] == 0) for i in 1:nrow(df_all)]
end

"===extra tree selection===" |> println
@time let
    trainX = df_all[1:length(trainY),:]      |> DataArray
    testX  = df_all[length(trainY)+1:end, :] |> DataArray

    model = skes.ExtraTreesClassifier(n_estimators = 100)
    model[:fit](trainX, trainY)

    importance = model[:feature_importances_]

    df_all = df_all[:,sortperm(importance)[1:200]]

    # writetable("feature.csv", DataFrame(
    #     feature = names(df_all)[sortperm(importance)],
    #     importance = importance[sortperm(importance)]
    # ))
end

"===saving as csv===" |> println
@time let
    df_train = df_all[1:length(trainY),:]
    df_test  = df_all[length(trainY)+1:end,:]
    df_train[:TARGET] = trainY

    writetable("train.processed.csv", df_train)
    writetable("test.processed.csv", df_test)
end

"===saving as jld===" |> println
@time let
    save("processed.jld", "trainX", df_all[1:length(trainY),:] |> DataArray)
    save("processed.jld", "testX", df_all[length(trainY)+1:end,:] |> DataArray)
    save("processed.jld", "trainY", trainY)
end
