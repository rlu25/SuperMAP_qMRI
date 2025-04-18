clear all;
directory_path = {'../dataset/Testing Data'};
file_filename = {'0118','0119','0123','0127','0128','0129','0139','0141','0144'};
image_filename = {'TB4','TB6','TB8'};
map_w_mask = zeros(320,320,216);
image_r4 = zeros(320,320,216,8);image_r6 = zeros(320,320,216,8);image_r8 = zeros(320,320,216,8);
for ii = 1:length(file_filename)

    close all;
    file_path1 = [directory_path{1},'/',file_filename{ii},'/retro/'];
    % file_path1 = [directory_path{1},'/',file_filename{ii},'/pros/'];
    ii
    loadfile_name = strcat(directory_path{1},'/',file_filename{ii},'/Ref.mat');
    mask_name = strcat(directory_path{1},'/',file_filename{ii},'/mask.mat');
    load(loadfile_name)
    load(mask_name)
        
    map(find(map>200))=199;
    map(isnan(map))=199;
    
    a=recon(:,:,:,1);
    % mask=(a>0.1);
    map_w_mask(:,:,(ii-1)*24+1:ii*24)=map.*mask;
    for iii = 1:length(image_filename)

        
            image_name = strcat(file_path1,image_filename{iii},'.mat');
            load(image_name)
            recon=abs(recon);
            recon=recon./max(recon(:));
            eval(['image_r',num2str(iii*2+2),'(:,:,(ii-1)*24+1:ii*24,:)=recon;']);
        
    end
end
save('test_retro_rf4.mat','image_r4','map_w_mask','-v7.3')
save('test_retro_rf6.mat','image_r6','map_w_mask','-v7.3')
save('test_retro_rf8.mat','image_r8','map_w_mask','-v7.3')

