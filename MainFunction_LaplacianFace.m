%%
% This is the main function of Laplacian Face algorithm
% Facial recognition using Laplacian eigenmap
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
image_row=96;
image_col=84;
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
%   temp_vector=train_image_set(100+a,:);
%   temp_image=(coeff_set.')*temp_vector.';
%   subplot(4,4,a),imshow(reshape(temp_image,image_row,image_col),[]);
%end

%%
% Build Laplacian model
% param contains additional variables 
% The original model uses the neighborhood of facial images
acc_neighbor=[];
for vector_num=1:200
    params1.neighbor_num=7;
    params1.norm='F-norm';
    params1.reduced_dim=dim;
    params1.batch_size=200;
    params1.image_num=train_image_num;
    params1.vector_num=vector_num;
    params1.precision='low';
    params1.distance_power=2e+05;
    [L1,L1_eigenvector,L1_eigenvalue]=LaplacianModel(train_image_set,'neighborhood',params1);
    %disp('Laplacian eigenmap completed, displaying some Laplacian Faces');
    %figure(1),title('First 9 Laplacian Faces Display');
    %for a=1:9
        %subplot(3,3,a),
        %imshow(reshape((coeff_set.')*L1_eigenvector(:,a),[image_row,image_col]),[]);
    %end

%%
% Classification section
    disp('Now testing the facial images');
    params2.train_image_num=train_image_num;
    params2.test_image_num=test_image_num;
    params2.image_row=image_row;
    params2.image_col=image_col;
    params2.vector_num=vector_num;
    params2.knn_mode='real';
    acc1=LaplacianClassify(train_image_set,train_label_set,test_image_set,test_label_set,L1_eigenvector,params2);
    fprintf('The accuracy of original model is %.3f\n',acc1);
    acc_neighbor=[acc_neighbor,acc1];
end

%%
% Build improved model
% The improved model uses high-dimensional adjacency ball of facial images
acc_ball=[];
for r=1.5e+03:1e+02:3e+03
    for vector_num=1:200   
        params3.ball_radius=r;
        params3.norm='F-norm';
        params3.reduced_dim=dim;
        params3.batch_size=200;
        params3.image_num=train_image_num;
        params3.vector_num=vector_num;
        params3.precision='low';
        params3.distance_power=r.^1.8;
        [L2,L2_eigenvector,L2_eigenvalue]=LaplacianModel(train_image_set,'adjacency_ball',params3);
        %disp('Laplacian eigenmap completed');
        %figure(1),title('First 9 Laplacian Faces Display');
        %for a=1:9
            %subplot(3,3,a),
            %imshow(reshape((coeff_set.')*L1_eigenvector(:,a),[image_row,image_col]),[]);
        %end
%%
% Classification Section
        disp('Now testing the facial images');
        params4.train_image_num=train_image_num;
        params4.test_image_num=test_image_num;
        params4.image_row=image_row;
        params4.image_col=image_col;
        params4.vector_num=vector_num;
        params4.knn_mode='real';
        acc2=LaplacianClassify(train_image_set,train_label_set,test_image_set,test_label_set,L2_eigenvector,params4);
        fprintf('The accuracy of original model is %.3f\n',acc2);
        acc_ball=[acc_ball,acc2];
    end
end
