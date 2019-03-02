%% Laplacian Classification
% image_set: facial images waiting for prediction
% label_set: labels for evaluation
% eigenvector,eigenvalues: variables of L_map, from LaplacianModel
% params: additional parameters
function accuracy=LaplacianClassify(train_image_set,train_label_set,test_image_set,test_label_set,eigenvector,params)
% calculate the coordinates of training images under projection of LaplacianFace
train_image_num=params.train_image_num;
test_image_num=params.test_image_num;
vector_num=params.vector_num;
knn_mode=params.knn_mode;

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

% because of the limit in the precision of eigen decomposition
% coordinates may include complex numbers
if strcmp(knn_mode,'real')
    % use real coordinates to generate KNN model
    knn=fitcknn(train_coordinate_set,train_label_set,'NumNeighbors',1,'Distance','cosine','Prior','uniform');
    predict_label_set=predict(knn,test_coordinate_set);
    accuracy=length(find(predict_label_set==test_label_set))/test_image_num;
elseif strcmp(knn_mode,'real-imag')
    % add the imaginary matrix following after real matrix, doubling the dimension of coordinates
    new_train_set=[real(train_coordinate_set),imag(train_coordinate_set)];
    new_test_set=[real(test_coordinate_set),imag(test_coordinate_set)];
    knn=fitcknn(new_train_set,train_label_set,'NumNeighbors',1,'Distance','cosine','Prior','uniform');
    predict_label_set=predict(knn,new_test_set);
    accuracy=length(find(predict_label_set==test_label_set))/test_image_num;
end
