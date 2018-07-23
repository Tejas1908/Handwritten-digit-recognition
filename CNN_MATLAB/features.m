figure(1)

for i=1:8
    for j=1:8
    subplot(8,8,8*(i-1)+j)
    colormap(gray)
    imagesc(net.Layers(10,1).Weights(:,:,i,j))
    end
end