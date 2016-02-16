using DataFrames
using OhMyJulia: nrow, ncol, @map

immutable Dispatcher{Name} end

factorize!(df_train::DataFrame, df_test::DataFrame) = begin
	factorize(col) = factorize(col, df_train[col])
	factorize{T}(col, x::DataArray{T,1}) = nothing
	factorize(col, x::DataArray{UTF8String,1}) = begin
		level = x |> unique |> dropna |> sort |> enumerate |> @map(x[2]=>x[1]) |> Dict{UTF8String, Int}
		if level.count < 2
			error("too less levels")
		elseif level.count == 2 || find(x->length(x)>1,[keys(level)...]) |> length == 0
			factorize(col, level, :level)
		elseif level.count in 3:100
			factorize(col, level, :one_hot_encoding)
		else
			factorize(col, level, :level)
		end
	end
	factorize(col, level, method::Symbol) = factorize(col, level, Dispatcher{method}())
	factorize(col, level, method::Dispatcher{:level}) = begin
		f(x) = get(level, x, NA)
		map(f,df_train[col]), map(f,df_test[col])
	end
	factorize(col, level, method::Dispatcher{:one_hot_encoding}) = begin
		f(x) = begin
			y = zeros(Int8, nrow(x), level.count) |> DataArray
			for (i,v) in x[col]|>enumerate
				if v in keys(level)
					y[i,level[v]] = 1
				else
					y[i,:] = NA
				end
			end
			y
		end
		f(df_train), f(df_test)
	end
	factorize(col, level, method::Dispatcher{:statistics_transformation}) = begin
		prop = Dict(level)
		# for i in keys(level)
		# 	appears = map(df_train[col]) do x isna(x) ? false : x == i end |> Array{Bool,1}
		# 	prop[i] = sum(df_train[:target][appears]) / sum(appears)
		# end
		for i in keys(level)
			prop[i] = 0
		end
		acc = Dict(prop)
		for (i,v) in df_train[col]|>enumerate
			if isna(v) continue end
			prop[v] += df_train[i,:target]
			acc[v] += 1
		end
		prop = prop |> Dict{UTF8String,Float64}
		for i in keys(level)
			prop[i] = prop[i] / acc[i]
		end
		factorize(col, prop, :level)
	end

	replace(col, ::Void) = nothing
	replace{T}(col, x::Tuple{DataArray{T,1},DataArray{T,1}}) = df_train[col],df_test[col] = x
	replace{T}(col, x::Tuple{DataArray{T,2},DataArray{T,2}}) = begin
		train,test = x
		delete!(df_train, col)
		delete!(df_test, col)
		for i in 1:ncol(train)
			c = Symbol("$(col)_$(i)")
			df_train[c], df_test[c] = train[:,i], test[:,i]
		end
	end

	for i in names(df_train)[3:end]
		replace(i, factorize(i))
	end
end

writelibsvm{T}(filename, feature::DataArray{T,2}, label::DataArray{Int64,1}) = writelibsvm(filename, feature, Nullable(label))
writelibsvm{T}(filename, feature::DataArray{T,2}, label::Nullable{DataArray{Int64,1}}=Nullable{DataArray{Int64,1}}()) = begin # use for loop because the associativity of `reduce` is not reliable
	open(filename, "w") do output
		for i in 1:nrow(feature)
			if !isnull(label)
				print(output, get(label)[i], " ")
			end
			for j in 1:ncol(feature)
				v = feature[i,j]
				if !isna(v)
					@printf(output, "%d:%.12f ", j, v)
				end
			end
			print(output, "\n")
		end
	end
end

"===reading===" |> println
gc()
@time begin
	df_train = readtable("train.csv")
	df_test = readtable("test.csv")
end

"===factorizing===" |> println
gc()
@time begin
	factorize!(df_train, df_test)
	writetable("dump.csv",df_train)
end

"===reformating===" |> println
gc()
@time begin
	xTrain = df_train[:,3:end] |> DataArray
	yTrain = df_train[:target] |> DataArray
	xTest  = df_test[:,2:end]  |> DataArray
	df_train = df_test = nothing
end

"===writing out===" |> println
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

#TODO: add ballast for statistics transformation
