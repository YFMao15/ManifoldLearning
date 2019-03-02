%%
% This is the main function of LTSA algorithm
% Facial recognition using Local Tangent Space Alignment Face
% Dataset: YaleFaceB, size 5MB
% Facial images are encoded in the format of pgm

clc
clear all
close all
%%
% Read images
% This file should be placed in the same directory with facial image folder 'YaleData'

folder='YaleFace Data';
fprintf('Read Images from %s\n',folder);
folder_content=dir(folder);
train_image_set=[];
test_image_set=[];
train_label_set=[];
test_label_set=[];
train_image_num=0;
test_image_num=0;
image_row=48;
image_col=42;
percentage=0.99;
% about 70% of all images consist of training set, the rest are testing set
ratio=0.7;
for count=1:length(folder_content)
    if length(folder_content(count).name)>2
        subfolder=folder_content(count).name;
        subfolder_content=dir([folder,'/',subfolder]);
        for sub_count=1:length(subfolder_content)
            image_path=[subfolder_content(sub_count).folder,'\',subfolder_content(sub_count).name];
            if contains(image_path,'.pgm')
                rng=rand();
                if rng<ratio
                    temp_image=imresize(round(255*im2double(imread(image_path))),[image_row,image_col]);
                    % imshow(temp_image,[]);
                    temp_image=reshape(temp_image,1,[]);              
                    train_image_set=[train_image_set;temp_image];
                    train_label_set=[train_label_set;count];
                    train_image_num=train_image_num+1;
                else
                    temp_image=imresize(round(255*im2double(imread(image_path))),[image_row,image_col]);
                    % imshow(temp_image,[]);
                    temp_image=reshape(temp_image,1,[]);              
                    test_image_set=[test_image_set;temp_image];
                    test_label_set=[test_label_set;count];
                    test_image_num=test_image_num+1;
                end
            end
        end
    end
end

[train_image_set,test_image_set,coeff_set,dim]=Preprocessing(train_image_set,test_image_set,train_image_num,test_image_num,percentage);

%disp('Display some faces after preprocessing');
%figure(1),
%for a=1:16
   %temp_vector=train_image_set(100+a,:);
   %temp_image=(coeff_set.')*temp_vector.';
   %subplot(4,4,a),imshow(reshape(temp_image,image_row,image_col),[]);
%end


%%
% Build LTSA model
% param contains additional variables
acc_fixed_threshold=[];
for neighbor_num=20
    for vector_num=1:100
        params1.image_num=train_image_num;
        params1.batch_size=200;
        params1.neighbor_num=neighbor_num;
        params1.vector_num=vector_num;
        params1.reduced_dim=dim;
        params1.norm='F-norm';
        [LTSA_vector1,LTSA_value1,vector_num]=LTSAModel(train_image_set,'fixed_threshold',params1);
        
%%
% Classification section
        disp('Now testing the facial images');
        params2.train_image_num=train_image_num;
        params2.test_image_num=test_image_num;
        params2.image_row=image_row;
        params2.image_col=image_col;
        params2.vector_num=vector_num;
        acc1=LTSAClassify(train_image_set,train_label_set,test_image_set,test_label_set,LTSA_vector1,params2);
        fprintf('The accuracy of original model is %.3f\n',acc1);
        acc_fixed_threshold=[acc_fixed_threshold,acc1];
    end
end

%%
% Build improved model
% cov_percentage controls the subspace components instead of vector_num
% which means the covariance of neighborhood 
acc_fixed_threshold=[];
for cov_percentage=0.95:0.01:0.99
    for vector_num=1:100
        params3.image_num=train_image_num;
        params3.batch_size=200;
        params3.neighbor_num=neighbor_num;
        params3.vector_num=vector_num;
        params3.cov_percentage=cov_percentage;
        params3.reduced_dim=dim;
        params3.norm='F-norm';
        [LTSA_vector1,LTSA_value1,vector_num]=LTSAModel(train_image_set,'neighbor_covariance',params3);
        
%%
% Classification section
        disp('Now testing the facial images');
        params4.train_image_num=train_image_num;
        params4.test_image_num=test_image_num;
        params4.image_row=image_row;
        params4.image_col=image_col;
        params4.vector_num=vector_num;
        acc1=LTSAClassify(train_image_set,train_label_set,test_image_set,test_label_set,LTSA_vector1,params4);
        fprintf('The accuracy of original model is %.3f\n',acc1);
        acc_fixed_threshold=[acc_fixed_threshold,acc1];
    end
end