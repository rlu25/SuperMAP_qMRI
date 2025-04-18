clear all;
directory_path = {'../dataset/Training Data/LI_STUDY0041','../dataset/Training Data/LI_STUDY0039','../dataset/Training Data/LI_STUDY0038_right','../dataset/Training Data/LI_STUDY0038_left'};
filenames = {'','_re'};
% filenames = {''};
image_filename = {'TB4','TB6','TB8'};
m = zeros(320,320,15*8);
map = zeros(320,320,15*8);
map_w_mask = zeros(320,320,15*8);
image_r4 = zeros(320,320,15*8,8);image_r6 = zeros(320,320,15*8,8);image_r8 = zeros(320,320,15*8,8);
for ii = 1:length(directory_path)
    
    close all;
    file_path1 = [directory_path{ii},'/'];
    file_path2 = [directory_path{ii},'/retro/'];
    
    for iii = 1:length(filenames)
	
        loadfile_name = strcat(file_path1,'Ref',filenames{iii},'.mat');
        mask_name = strcat(file_path1,'mask',filenames{iii},'.mat');
        load(loadfile_name)
        load(mask_name)
        % k = ii
        k = (ii-1)*2+iii
        T1rho(find(T1rho>200))=199.9;
        T1rho(isnan(T1rho))=199.9;
        map(:,:,(k-1)*15+1:k*15)=T1rho(:,:,5:19);
        a=recon(:,:,:,1);
        mask=(a>0.1);
        m(:,:,(k-1)*15+1:k*15)=mask(:,:,5:19);
        map_w_mask(:,:,(k-1)*15+1:k*15)=T1rho(:,:,5:19).*mask(:,:,5:19);
        
        for iiii = 1:length(image_filename)
            image_name = strcat(file_path2,image_filename{iiii},'',filenames{iii},'.mat');
            load(image_name)
            recon=abs(recon);
            recon=recon./max(recon(:));
            eval(['image_r',num2str(iiii*2+2),'(:,:,(k-1)*15+1:k*15,:)=recon(:,:,5:19,:);']);
        end
    end
end
save('train_rf4.mat','map_w_mask','m','map','image_r4','-v7.3')
save('train_rf6.mat','map_w_mask','m','map','image_r6','-v7.3')
save('train_rf8.mat','map_w_mask','m','map','image_r8','-v7.3')