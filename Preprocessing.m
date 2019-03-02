%% PCA Preprocessing
% Use PCA to obtain main components and avoid singularity in subspace projection
% Also preprocess all images by reducing the dimensionality
% train_image_set,test_image_set: input original images
% train_image_num,test_image_num: number of images
% percentage: the ratio of information preserved in the facial images
function [out_train_set,out_test_set,coeff_set,dim]=Preprocessing(train_image_set,test_image_set,train_image_num,test_image_num,percentage)
image_set=[train_image_set;test_image_set];
[coeff,~,varlist]=pca(image_set);
% obtain main components
total_var=sum(varlist);
var_percentage=0;
dim=0;
while var_percentage<percentage
    var_percentage=sum(varlist(1:dim))/total_var;
    dim=dim+1;
end
coeff_set=(coeff(:,1:dim)).';
inv_coeff_set=(coeff_set*(coeff_set.'))^(-1);
% use the selected coeffcients to reduce dimensionality
output_set=(inv_coeff_set*coeff_set*image_set.').';
out_train_set=output_set(1:train_image_num,:);
out_test_set=output_set(train_image_num+1:train_image_num+test_image_num,:);