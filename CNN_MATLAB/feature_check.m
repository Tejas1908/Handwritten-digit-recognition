input_image=X_train(:,:,1,1);
figure(1)
colormap(gray)
imagesc(input_image);

conv_map_1=[];
relu_1=[];

figure(2)
for i=1:8
    convolved_image=conv2(input_image,net.Layers(2,1).Weights(:,:,1,i),'valid');
    conv_map_1=cat(3,conv_map_1,convolved_image);
    relu_1=cat(3,relu_1,max(0,convolved_image(:,:)));
    subplot(8,1,i)
    colormap(gray)
    imagesc(conv_map_1(:,:,i));
end

figure(3)
for i=1:8
    subplot(8,1,i)
    colormap(gray)
    imagesc(relu_1(:,:,i));
end

conv_map_2=[];
relu_2=[];


for i=1:16
    convolved_image=convn(relu_1,net.Layers(6,1).Weights(:,:,:,i),'valid');
    conv_map_2=cat(3,conv_map_2,convolved_image);
    relu_2=cat(3,relu_2,max(0,convolved_image(:,:)));
    
end

figure(4)
for i=1:8
    subplot(8,1,i)
    colormap(gray)
    imagesc(relu_2(:,:,i));
end

conv_map_3=[];
relu_3=[];


for i=1:32
    convolved_image=convn(relu_2,net.Layers(10,1).Weights(:,:,:,i),'valid');
    conv_map_3=cat(3,conv_map_3,convolved_image);
    relu_3=cat(3,relu_3,max(0,convolved_image(:,:)));
    
end

figure(5)
for i=1:8
    subplot(8,1,i)
    colormap(gray)
    imagesc(relu_3(:,:,i));
end
