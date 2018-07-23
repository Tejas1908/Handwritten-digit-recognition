X_test=testX';
y_test=testY';

[probability,label]=max(net(X_test));

num_correct=nnz(label'==y_test);
accuracy=num_correct/numel(y_test)*100
