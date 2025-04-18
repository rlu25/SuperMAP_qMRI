clear all;
load rf8_retro.mat
load test_rf8_retro_all.mat
file_filename = {'0118','0119','0123','0127','0128','0129','0139','0141','0144'};
n=0;
for count=1:9

[nslice,nc,nx,ny]=size(image_mat);
map=zeros(nx*8,ny*8,24);
ref=zeros(nx*8,ny*8,24);

nx=nx-1;
for i=1:24
    for j=1:8
        for k=1:8
            n=n+1;
            map((j-1)*nx+1:j*40,(k-1)*nx+1:k*40,i)=squeeze(image_mat(n,1,1:40,1:40));
            ref((j-1)*nx+1:j*40,(k-1)*nx+1:k*40,i)=squeeze(test(n,5,1:40,1:40));
        end
    end
end
m=(ref>0);
% map=map.*m;
map_w_mask=map.*m;
figure;imagesc(map_w_mask(1:320,1:320,10)*200,[0 70]);colorbar
figure;imagesc(ref(1:320,1:320,10)*200,[0 70]);colorbar
savename=strcat(file_filename{count},'_retro_8.mat');
save(savename,'map','map_w_mask','-v7.3')
end