### preparing ###
using DataFrames
using XGBoost
using DecisionTree
using OhMyJulia: nrow

### read data ###
df_population = readtable("age_gender_bkts.csv")
df_country = readtable("countries.csv")
df_train = readtable("train_users.csv")
df_test = readtable("test_users.csv")
#df_session = readtable("sessions.csv")
nTrain = df_train |> nrow
nTest = df_test |> nrow
nAll = nTrain + nTest

### basic process ###
delete!(df_population, :year) # as all years are 2015
yTrain = df_train[:country_destination] |> Array
df_all = vcat(df_train[:,1:end-1], df_test)
df_train = df_test = nothing # free memory

### features
features = UTF8String[]

# one hot encoding, for missing values, all features are 0
one_hot_encode(colname) = begin
	states = df_all[colname] |> unique
	states = states[~isna(states)] # i dont know if there a more elegent way to filter NAs. it seems that built-in "Base.filter" return strange things i dont want
	code = zeros(nAll,length(states))

	for i in 1:nAll
		j = df_all[i, colname]
		if ~isna(j)
			code[i,findfirst(states, j)] = 1.0
		end
	end

	for i in states
		push!(features, "$colname: $i")
	end

	code
end

oneHotEncodedFeatures = map(one_hot_encode, [:gender,
	:signup_method, :language, :affiliate_channel,
	:affiliate_provider, :first_affiliate_tracked,
	:signup_app, :first_device_type, :first_browser])

xAll = hcat(oneHotEncodedFeatures...)
xTrain = xAll[1:nTrain,:]
xTest = xAll[nTrain+1:end,:]
xAll = nothing

### train and fit ###
#xgb
#num_round = 2
#nfold = 4
#param = Dict("max_depth"=>2, "eta"=>1, "objective"=>"binary:logistic")
#nfold_cv(xTrain, num_round, nfold, label=yTrain, param=param)
#model = xgboost(xTrain, 2, label=yTrain, eta=1, max_depth=2)
#pred = predict(model, xTest)

#random forest
#model = build_forest(yTrain, xTrain, 50, 10, 0.5)
#apply_forest(model, xTest)
accuracy = nfoldCV_forest(yTrain, xTrain, 50, 10, 3, 0.5)

# Ideas:
# 1. replace age<14 or age>100 with avrage
# 2. whether first act before register
# 3. 
