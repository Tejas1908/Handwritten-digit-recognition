

[probability,YPred] = max(predict(net,X_test)');
YPred=YPred-1;
num_correct = nnz(YPred == y_test);
accuracy=num_correct/numel(y_test)*100



