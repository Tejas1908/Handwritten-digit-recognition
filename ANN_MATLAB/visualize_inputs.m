figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:36                                    % preview first 36 samples
    subplot(6,6,i)                              % plot them in 6 x 6 grid
    digit = reshape(trainX(i, :), [28,28]);
    digit=digit';                               % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(trainY(i)))                   % show the label
end