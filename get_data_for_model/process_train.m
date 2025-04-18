clear;close all;clc;
directory_path = {'../get_data/'};
filenames = {'train_rf8'};

size_input = 41; size_label = 41; stride = 4; batchsize= 64;

%==========================================================================
loadfile_name = strcat(directory_path{1},filenames{1},'.mat');
load(loadfile_name);                                                 
input = image_r8;   % 320   320   192     8

%input=abs(input);
%choose echo
selected_4 = [1,2,4,8];         input = input(:,:,:,selected_4);

max_Input = max(input(:));  min_Input = min(input(:)); mean_Input = mean(input(:));
%========================================================================== %disp(['xxx:' num2str(xxx)]);
target = map_w_mask; % 320   320   192
target = abs(target);
max_ref = max(target(:));
min_ref = min(target(:));
mean_ref = mean(target(:));
                
target = target./200;

the_input = input;            size(the_input);
the_label = target;           size(the_label);
 
%==========================================================================
data = zeros(15*8*70*70, 5, size_input, size_input); %label = zeros(size_label, size_label, 1, 24*7*19600); 
padding = abs(size_input - size_label)/2; 
count = 0;
a=0;
for slice = 1 : 15*8                                                  
    slice
    
    %area1 = 11:310; area2 = 51:300;
    
    im_input = squeeze(the_input(:,:,slice,:));     
    im_label = the_label(:,:,slice,:);  
    
    subplot(1,5,1);imshow(abs(im_input(:,:,1)),[]);title([' Input Slice: ',num2str(slice)]);
    subplot(1,5,2);imshow(abs(im_input(:,:,2)),[]);title([' Input Slice: ',num2str(slice)]);
    subplot(1,5,3);imshow(abs(im_input(:,:,3)),[]);title([' Input Slice: ',num2str(slice)]);
    subplot(1,5,4);imshow(abs(im_input(:,:,4)),[]);title([' Input Slice: ',num2str(slice)]);
    subplot(1,5,5);imshow(abs(im_label(:,:)),[]);title('Map'); pause(0.3);
    [hei,wid,gao] = size(im_label);
    
       for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1
                a=a+1;
                subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
                subim_input = permute(subim_input, [3 1 2]);
                subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
                if (min(subim_label(:) == 0))
                   
                else
                    count = count+1;
                    data(count, 1:4, :, :) = subim_input;
                    data(count, 5, :, :) = subim_label;
                end
            end
        end
end
data(count+1:a,:,:,:)=[];
save('train_data_rf8.mat','data','-v7.3')