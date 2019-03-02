%% Linear Tangent Subpace Alignment Classification
% image_set: facial images waiting for prediction
% label_set: labels for evaluation
% eigenvector,eigenvalues: variables of LTSA_map, from LTSAModel
% params: additional parameters
function accuracy=LTSAClassify(train_image_set,train_label_set,test_image_set,test_label_set,eigenvector,params)
% calculate the coordinates of training images under projection of LaplacianFace
train_image_num=params.train_image_num;
test_image_num=params.test_image_num;
vector_num=params.vector_num;

train_coordinate_set=zeros(train_image_num,vector_num);
test_coordinate_set=zeros(test_image_num,vector_num);
inv_matrix=(eigenvector'*eigenvector)^(-1);
for a=1:train_image_num
    temp_image=train_image_set(a,:);
    train_coordinate_set(a,:)=(inv_matrix*eigenvector.'*temp_image.').';
end
for a=1:test_image_num
    temp_image=test_image_set(a,:);
    test_coordinate_set(a,:)=(inv_matrix*eigenvector.'*temp_image.').';
end

% use real coordinates to generate KNN model
knn=fitcknn(train_coordinate_set,train_label_set,'NumNeighbors',1,'Distance','cosine','Prior','uniform');
predict_label_set=predict(knn,test_coordinate_set);
accuracy=length(find(predict_label_set==test_label_set))/test_image_num;

