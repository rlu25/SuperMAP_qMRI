clear;close all;clc;
directory_path = {'../get_data/'};
filenames = {'test_retro_rf8'};

size_input = 41; size_label = 41; stride = 40; batchsize= 64;

%==========================================================================
loadfile_name = strcat(directory_path{1},filenames{1},'.mat');
load(loadfile_name);                                                 
input = image_r8;   % 320   320   192     8
% file_filename = {'0118','0119','0123','0127','0128','0129','0139','0141','0144'};
%input=abs(input);

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
% for i=1:length(file_filename)
%     savefile_name = strcat('test_',file_filename{i},'_rf4retro.mat'); 

test = zeros(size_input, size_input, 5, 1536); %label = zeros(size_label, size_label, 1, 24*7*19600); 
padding = abs(size_input - size_label)/2; 
count = 0;
a=0;

for slice = 1 : 216                                              
    slice
    
    %area1 = 11:310; area2 = 51:300;
    
    im_input = squeeze(the_input(:,:,slice,:)); 
    im_input(321,:,:)=0;
    im_input(:,321,:)=0;
    im_label = the_label(:,:,slice,:);  
    im_label(321,:,:)=0;
    im_label(:,321,:)=0;
    % subplot(1,5,1);imshow(abs(im_input(:,:,1)),[]);title([' Input Slice: ',num2str(slice)]);
    % subplot(1,5,2);imshow(abs(im_input(:,:,2)),[]);title([' Input Slice: ',num2str(slice)]);
    % subplot(1,5,3);imshow(abs(im_input(:,:,3)),[]);title([' Input Slice: ',num2str(slice)]);
    % subplot(1,5,4);imshow(abs(im_input(:,:,4)),[]);title([' Input Slice: ',num2str(slice)]);
    % subplot(1,5,5);imshow(abs(im_label(:,:)),[]);title('Map'); pause(0.3);
    [hei,wid,gao] = size(im_label);
    
       for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1
                
                subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
                subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
                
                count = count+1;
                test(:, :, 1:4, count) = subim_input;
                test(:, :, 5, count) = subim_label;
               
            end
        end
end
test=permute(test,[4 3 1 2]);
save('test_rf8_retro_all.mat','test','-v7.3')
% end