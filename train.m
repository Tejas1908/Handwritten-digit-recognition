X = trainX';
%y = dummyvar(trainY)';

trainFcn = 'trainscg';                          % use scaled conjugate gradient for training
                                                %have other choices

hiddenLayerSize = 100;                          %other choices
net = patternnet(hiddenLayerSize);             

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'mse';                % performance function

net.performParam.regularization=0.4;            % regularization parameter

net.trainParam.max_fail=10;                     % train parameter validation checks
%net.trainParam.epochs=500;
%net.trainParam.lambda=500;                     % dont know what this does

[net,tr] = train(net,X,y);                      % train
