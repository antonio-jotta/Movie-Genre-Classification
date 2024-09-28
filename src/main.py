import mgc_input
import mgc_train
import mgc_test
import mgc_output

train, test = mgc_input.read_data()
mgc_train.train_model(train) # Uncomment if need to train again
preds = mgc_test.test_sample(test)
mgc_output.print_result(test, preds, train['Genre'].unique())