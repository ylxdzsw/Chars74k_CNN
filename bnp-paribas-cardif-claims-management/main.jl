using DataFrames
using OhMyJulia: nrow, ncol, @map

factorize{T}(x::DataArray{T,1}) = x
factorize(x::DataArray{UTF8String,1}) = begin
	d = x |> unique |> enumerate |> @map(x[2]=>x[1]) |> Dict
	map(x) do x isna(x) ? x : d[x] end
end
factorize!(x::DataFrame) = for i in x.colindex.names
	x[i] = x[i] |> factorize
end

writelibsvm{T}(filename, feature::DataArray{T,2}, label::DataArray{Int64,1}) = writelibsvm(filename, feature, Nullable(label))
writelibsvm{T}(filename, feature::DataArray{T,2}, label::Nullable{DataArray{Int64,1}}=Nullable{DataArray{Int64,1}}()) = begin # use for loop because the associativity of `reduce` is not reliable
	output = open(filename, "w")
	for i in 1:nrow(feature)
		if !isnull(label)
			print(output, get(label)[i], " ")
		end
		for j in 1:ncol(feature)
			v = feature[i,j]
			if !isna(v)
				print(output, j, ":", v, " ")
			end
		end
		print(output, "\n")
	end
	close(output)
end

"===reading===" |> println
gc()
@time begin
	df_train = readtable("train.csv")
	df_test = readtable("test.csv")
	delete!(df_train, :ID)
	delete!(df_test, :ID)
end

"===factorizing===" |> println
gc()
@time begin
	factorize!(df_train)
	factorize!(df_test)
end

"===reformating===" |> println
gc()
@time begin
	xTrain = df_train[df_train|>names .!= :target] |> DataArray
	yTrain = df_train[:target] |> DataArray
	xTest = df_test |> DataArray
end

"===writing out===" |> println
df_train = df_test = nothing
gc()
@time begin
	writelibsvm("xgboost.train", xTrain, yTrain)
	writelibsvm("xgboost.test", xTest)
end

"===xgboosting===" |> println
gc()
@time begin
	run(`xgboost train.conf`)
	run(`xgboost test.conf`)
end

"===submitting===" |> println
gc()
@time begin
	result = readcsv("pred.txt")
	submit = readtable("sample_submission.csv")
	submit[:PredictedProb] = result[:,1]
	writetable("submit.csv", submit)
end




