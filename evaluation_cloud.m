%% This mfile reads the predicted images obtained by the activation function 
% of the last layer in a CNN (for instance, the output of a sigmoid). Then, it
% stitches those patch masks up together to create a mask for a complete Landsat
% 8 scene, and at last, it calculates the quantitative evaluators including
% Jaccard index, precision, recall, specificity, and accuracy in %.

% Author: Sorour Mohajerani
% Simon Fraser University, Canada
% email: smohajer@sfu.ca
% First version: 11 May 2017; This version: v1.5 (27 July 2019)

%%
% **** Change the paths ****
% Windows:
gt_folder_path = ('C:\Users\joaof\Dev\Papers-temp\2025-cloud-ccis-visapp\38-cloud\38-Cloud_test\Entire_scene_gts');
preds_folder_root = ('C:\Users\joaof\Dev\Papers-temp\2025-cloud-ccis-visapp\exp_38cloud');
% DISP
disp(gt_folder_path)
disp(preds_folder_root)

% Obter todas as pastas em "pred_folder_root"
conteudo = dir(preds_folder_root);

% Filtrar apenas as pastas, excluindo '.' e '..'
pastas = conteudo([conteudo.isdir]); 
preds_folder_list = {pastas.name}; 
preds_folder_list = preds_folder_list(~ismember(preds_folder_list, {'.', '..'}));
% Filtrar apenas as que começam com 'exp_'
preds_folder_list = preds_folder_list(startsWith(preds_folder_list, 'exp_'));
% DISP
%%% disp(preds_folder_list);
for i = 1:length(preds_folder_list)
    disp(preds_folder_list{i})
end

% **** Uncoment the experiments folder ****
% preds_folder_test = {'exp_Linknet_efficientnet-b2_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Linknet_mobilenet_v2_imagenet_24_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Linknet_resnet50_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Linknet_vgg16_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_PSPNet_efficientnet-b2_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_PSPNet_mobilenet_v2_imagenet_24_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_PSPNet_resnet50_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_PSPNet_vgg16_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Unet_efficientnet-b2_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Unet_mobilenet_v2_imagenet_24_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Unet_resnet50_imagenet_16_0.0001_1000_plateau_1\test_pred'};
% preds_folder = {'exp_Unet_vgg16_imagenet_16_0.0001_1000_plateau_1\test_pred'};

% DISP
% disp(preds_folder_test)

pr_patch_size_rows = 384;
pr_patch_size_cols = 384;

%"0" for clear and "1" for cloud
classes = [0, 1]; 

% Should be 1 for printing out the confusion matrix of each scene 
conf_matrix_print_out = 0; 

%Threshold for binarizing the output of the network and creating a binary 
% mask. Should be between [0,1].
thresh = 12 / 255; % The threshold used in Cloud-Net

for i = 1:length(preds_folder_list)
    disp('\n==================================================')
    preds_folder = {fullfile(preds_folder_list{i}, 'test_pred')};
    disp(preds_folder);

    % Getting unique sceneids existing in the presiction folder
    all_uniq_sceneid = extract_unique_sceneids(preds_folder_root, preds_folder);
    % fprintf('%s\n', all_uniq_sceneid)

    %% The patch masks are put together to generate a complete scene mask
    QE = [];
    scene_assess = {};
    for n = 1:length(all_uniq_sceneid)
        if n == length(all_uniq_sceneid)
            fprintf('Working on sceneID #%d: %s \n\n', n, char(all_uniq_sceneid(n,1)));
        else
            fprintf('Working on sceneID #%d: %s ... \n', n, char(all_uniq_sceneid(n,1)));
        end

        % JOAO (1) >>>>
        %disp(n);
        %disp(all_uniq_sceneid(n,1))

        % temp = strcat('edited_corrected_gts_', all_uniq_sceneid(n,1),'.TIF');
        % fprintf('%s\n', char(temp));
        % JOAO (1) <<<< 

        gt = imread(char(fullfile(gt_folder_path, strcat('edited_corrected_gts_', all_uniq_sceneid(n,1),'.TIF'))));

        % Finding all the patch masks corresponding to a unique sceneID
        % based on the sceneID from the name string of each patch mask.
        scid_related_patches = get_patches_for_sceneid(preds_folder_root, preds_folder, all_uniq_sceneid(n,1));

        % Generating a complete scene mask from the found patch masks
        complete_pred_mask = false();   
        for pcount = 1:length(scid_related_patches)
            predicted_patch_path =(fullfile (preds_folder_root, preds_folder, scid_related_patches(pcount,1)));

            % **** Carrega a imagem predita ****
            predicted_patch = imread(char(predicted_patch_path));

            % **** Remove canais que estão sobrando ***
            %%% predicted_patch = predicted_patch(:,:,1);

            [w, h] = size(predicted_patch);

            if w ~= pr_patch_size_rows || h ~= pr_patch_size_cols
            % resize to 384*384 in the case of predictions with different size from 384*384
                predicted_patch = imresize(predicted_patch, [pr_patch_size_rows ,pr_patch_size_cols]);
            end

            predicted_patch =  imbinarize(predicted_patch,thresh);

            raw_result_patch_name = char(scid_related_patches(pcount,1));

            % Getting row and column number of each patch from the name string of
            % each patch mask. this row and column obtain the location o each
            % patch mask in the complete mask.
            [patch_row , patch_col] = extract_rowcol_each_patch (raw_result_patch_name);

            % Stiching up patch masks together
            complete_pred_mask ((patch_row-1)*pr_patch_size_rows+1:(patch_row)... 
                                *pr_patch_size_rows,(patch_col-1)*pr_patch_size_cols+1:...
                                (patch_col)*pr_patch_size_cols ) = predicted_patch;
        end

        % Removing the zero padded distance around the whole mask
        complete_pred_mask = unzeropad (complete_pred_mask, gt);

        % Saving complete scene predicted masks    
        complete_folder = strcat('entire_masks_',char(preds_folder));
        if 7~=exist(fullfile(preds_folder_root ,complete_folder), 'dir')
            mkdir(preds_folder_root, complete_folder);
        end

        baseFileName = sprintf('%s.TIF', char(all_uniq_sceneid(n,1)));
        path = fullfile(preds_folder_root, complete_folder, baseFileName); 

        % **** Grava a imagem predita COMPLETA em disco ****
        imwrite(complete_pred_mask, path); 

        % Calculating the quantitative evaluators
        QE(n,:) = 100 .* QE_calcul(complete_pred_mask, gt, classes, conf_matrix_print_out);

        % Preparing evaluators for further saving in excel and txt files 
        scene_assess(n,:) = [all_uniq_sceneid(n,1), num2str(thresh), num2str(QE(n,1)) , ...
            num2str(QE(n,2)), num2str(QE(n,3)), num2str(QE(n,4)), num2str(QE(n,5))];
    end

    % Averaging evaluators over 20 landsat 8 scenes
    mean_on_test_data = mean(QE);

    fprintf('Average evaluators over %d scenes are: \n\n', n);
    fprintf('Precision, Recall, Specificity, Jaccard, Accuracy \n');
    fprintf(' %2.3f , %2.3f , %2.3f , %2.3f , %2.3f \n', ...
        mean_on_test_data(1,1), mean_on_test_data(1,2), mean_on_test_data(1,3), mean_on_test_data(1,4), mean_on_test_data(1,5));

    %% Saving the evaluators in a excel file
    x = strsplit(complete_folder, '\');
    excel_baseFileName = strcat('numerical_results_', x{1},'.xlsx');
    excelpath = fullfile(preds_folder_root, excel_baseFileName);

    % JOAO - Erro ao gravar no path original. Gravando na mesma pasta do arquivo .m.
    disp(excelpath)
    %excelpath = excel_baseFileName;
    %disp(excelpath)

    xlswrite(excelpath,{'Scene ID', 'Threshold', 'Precision', 'Recall', 'Specificity', 'Jaccard', 'Accuracy',}, 'sheet1', 'A1:G1');        
    position1 = strcat('A',num2str(2)); 
    position2 = strcat('G',num2str(n+1)); 
    position = strcat(position1,':',position2);
    xlswrite(excelpath, scene_assess, 'sheet1', position);

    %% Saving the average of evaluators in a text file
    txt_baseFileName = strcat('numerical_results_', x{1},'.txt');
    txtpath = fullfile(preds_folder_root, txt_baseFileName);
    fileID = fopen(txtpath, 'w');
    fprintf(fileID,'Threshold = %3f \r\n', thresh);
    fprintf(fileID,'Precision, Recall, Specificity, Jaccard, Overall Accuracy \r\n');
    fprintf(fileID,'%2.6f,  %2.6f, %2.6f, %2.6f, %2.6f\r\n', mean_on_test_data);
    fclose(fileID);
    
end

disp('Done!')

%% This function extracts the unique scene IDs in the prediction folder
function uniq_sceneid = extract_unique_sceneids(result_root, preds_dir)
    path_4landtype = fullfile (result_root, preds_dir );
    folders_inside_landtype = dir(char(path_4landtype));
    l1 = length(folders_inside_landtype); % number of the patch masks inside the pred flder
    
    sceneid_lists = {}; 
    for iix = 1:l1-2 
        raw_result_patch_name = char(folders_inside_landtype(iix+2,1).name);
        raw_result_patch_name = strrep(raw_result_patch_name, '.TIF', '');    
        str = {raw_result_patch_name};
        loc1 = cell2mat (regexp(str, 'LC'));
        leng = length(raw_result_patch_name);
        sceneid_lists(iix,1) = {raw_result_patch_name(loc1:leng)}; %this gets scene ID
    end
    uniq_sceneid = unique(sceneid_lists(3:end));
    
end
%% This function finds the row and column number present in the name of each patch
function [row, col] = extract_rowcol_each_patch (name)
        % (IMPORTANTE)
        name = strrep(name, '.TIF', '');  
        str = {name};
        loc1 = cell2mat (regexp(str, 'LC'));
        loc2 = cell2mat (regexp(str, 'h_'));
        patchbad = name(loc2+2:loc1-2);
        str2 = {patchbad};
        loc3 = cell2mat(regexp(str2,'_'));

        row = str2double(patchbad(loc3(1)+1:loc3(2)-1));
        col = str2double(patchbad(loc3(3)+1:end));
end

%% This function finds all the patch masks corresponding to a unique sceneID
% based on the sceneID from the name string of each patch mask.
function  related_patches = get_patches_for_sceneid (result_root, preds_dir, sceneid)
    path_4preds = fullfile(result_root, preds_dir);
    files_inside = dir(char(path_4preds));
    ps = contains({files_inside(3:end,1).name},sceneid);
    desired_rownums = find(ps==1);
    le = length(desired_rownums); 
    
    related_patches={};
    for nsp = 1: le
        related_patches(nsp,1) = {char(files_inside(desired_rownums(1,nsp)+2,1).name)};
    end
end

%% This function removes the zero pads around a complete scene mask.
% This  padding had been added to each scene before cropping it to 
% small patches
function out = unzeropad (in_dest, in_source)
    [ny, nx] = size(in_dest); 
    [nys, nxs] = size (in_source);
    tmpy= floor((ny-nys)/2);
    tmpx= floor((nx-nxs)/2);
    
    out = in_dest(tmpy+1:tmpy+nys, tmpx+1:tmpx+nxs);
end

%% This prepares the data for calculating the evaluators.
function out = QE_calcul (predict,gt, labels, conf_print)
    out_both = cfmatrix(gt(:), predict(:), labels, 1, conf_print);
    
    % We are interested in getting the numerical evaluators of "1" or
    % "cloud" class only
    ix = find(labels(:)==1);
    out = out_both (ix,:); 
end
