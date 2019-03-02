%% LSTA Model for facial recognition
% image_set: input facial subspace in vectorized images
% mode: the way to calculate Laplacian maps 
% params: additional variables
function [LTSA_vector,LTSA_value,vector_num]=LTSAModel(image_set,mode,params)
image_num=params.image_num;
batch_size=params.batch_size;
neighbor_num=params.neighbor_num;
vector_num=params.vector_num;
reduced_dim=params.reduced_dim;
norm=params.norm;

aver_image_set=sum(image_set,1);
center_image_set=image_set-repmat(aver_image_set,image_num,1);
opts.maxit=1000;
opts.tol=1e-15;
% determine the neighborhood matrix of each image vector
% batch processing to prevent memory depletion
index_matrix=[];
for a=1:batch_size:image_num
    top_num=a+batch_size-1;
    bottom_num=a;
    if top_num>image_num
        top_num=image_num;
    end
    batch=image_set(bottom_num:top_num,:);
    if norm=='F-norm'
        temp1=sum(batch.'.*batch.');
        temp2=sum(image_set.'.*image_set.');
        temp3=batch*image_set.'; 
        batch_distance=real(sqrt(repmat(temp1.',[1,size(temp2,2)])+repmat(temp2,[size(temp1,2),1])-2*temp3));    
        [~,index]=sort(batch_distance,2);
        index_matrix=[index_matrix;index];
    end
end

if strcmp(mode,'fixed_threshold')
    vector_num=params.vector_num;
    % compute vector_num largest eignevectors of correlation matrix of each neighboring matrix
    % add them together to form the subspace matrix
    subspace_matrix=zeros(reduced_dim);
    % options for eigenvalue decomposition
    opts.maxit=1000;
    opts.tol=1e-15;
    for a=1:image_num
        % neighborhood includes itself
        neighbor_matrix=image_set(index_matrix(a,1:neighbor_num),:);
        mean=sum(neighbor_matrix)/neighbor_num;   
        % here the eigenvalue of centered covariance matrix is PCA component variance
        [coeff,~,~]=eigs((neighbor_matrix-mean).'*(neighbor_matrix-mean),vector_num,'lm',opts);
        extend_coeff=[1/sqrt(neighbor_num)*ones(reduced_dim),coeff];
        subspace_matrix=subspace_matrix+eye(reduced_dim)-extend_coeff*extend_coeff.';
    end

    % calculate the smallest eigenvector of subspace matrix to obtain new coordinates
    [temp_vector,temp_value]=eigs(subspace_matrix,vector_num,'sm',opts);
    LTSA_vector=temp_vector(:,1:vector_num);
    LTSA_value=temp_value(:,1:vector_num);
    
elseif strcmp(mode,'neighbor_covariance')
    cov_percentage=params.cov_percentage;
    subspace_matrix=zeros(reduced_dim);
    % options for eigenvalue decomposition
    opts.maxit=1000;
    opts.tol=1e-15;
    for a=1:image_num
        % neighborhood includes itself
        count=1;
        neighbor_matrix=image_set(index_matrix(a,1:neighbor_num),:);
        mean=sum(neighbor_matrix)/neighbor_num;   
        % here the eigenvalue of centered covariance matrix is PCA component variance
        [coeff,varlist,~]=eigs((neighbor_matrix-mean).'*(neighbor_matrix-mean),floor(reduced_dim/2),'lm',opts);
        while total_cov<cov_percentage*sum(varlist)
            count=count+1;
        extend_coeff=[1/sqrt(neighbor_num)*ones(reduced_dim),coeff(:,1:count)];
        subspace_matrix=subspace_matrix+eye(reduced_dim)-extend_coeff*extend_coeff.';
        end
    end

    % calculate the smallest eigenvector of subspace matrix to obtain new coordinates
    [temp_vector,temp_value]=eigs(subspace_matrix,vector_num,'sm',opts);
    LTSA_vector=temp_vector(:,1:vector_num);
    LTSA_value=temp_value(:,1:vector_num);
end