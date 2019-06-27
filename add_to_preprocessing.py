# make distinct train, dev, test sets
# this is just what we did for train with the names changed
# get rating for each video in each set
dev_text = []
dev_labels = []
for file_name in dev:
    #print(file_name)
    a_text,a_acoustic,a_visual,a_y_label = read_a_file(file_name)
    dev_text.append(a_text)
    dev_labels.append(a_y_label)
    #print(a_text,a_acoustic,a_visual,a_y_label)

test_text = []
test_labels = []
for file_name in test:
    #print(file_name)
    a_text,a_acoustic,a_visual,a_y_label = read_a_file(file_name)
    test_text.append(a_text)
    test_labels.append(a_y_label)
    #print(a_text,a_acoustic,a_visual,a_y_label)

# repeat matrix creation and flattening processes with these sets

# save matrices
data_dict = {"train":[train_vector, train_labels], "dev":[dev_vector, dev_labels], "test":[test_vector, test_labels]}
with open("vectors.pkl", "wb") as f:
    pickle.dump(data_dict)

# run svr on train set
all_data = load_pickle(os.path.join(data_path, "vectors.pkl"))
train_data = all_data["train"]  # can use data_dict instead
dev_data = all_data["dev"]
test_data = all_data["test"]
model = SVR(train_data[0].shape[0], d_text)  # from sklearn.svm import SVR
model.fit(train_data[0], [train_data][1])

# cross-validate with dev
# from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, dev_data[0], dev_data[1]) # can customize scoring method
avg_fit_time = np.mean(cv_results["fit_time"])
avg_score_time = np.mean(cv_results["score_time"])
avg_score = np.mean(cv_results["test_score"])

# get other stats later, like stdev and median
test_results = model.predict(test_data[0])
test_score = model.score(test_data[0], test_results)

# report final accuracy
print("Cross-validation results: " + cv_results)
print("Test score: " + test_score)
