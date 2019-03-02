%% Laplacian model
% image_set: input facial subspace in vectorized images
% mode: the way to calculate Laplacian maps 
% params: additional variables
function [L,L_eigenvector,L_eigenvalue]=LaplacianModel(image_set,mode,params)
if strcmp(mode,'neighborhood')
    % neighrborhood mode, which uses linear combination of neighboring elements
    disp('Using neighborhood to build Laplacian eigenmap');
    neighbor_num=params.neighbor_num;
    norm=params.norm;
    batch_size=params.batch_size;
    image_num=params.image_num; 
    vector_num=params.vector_num;
    dt_matrix=zeros(image_num,image_num);
    
    % batch processing to prevent memory depletion
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
            % use the traditional way to test the fast algorithm of distance calculation
            %batch_distance_test=zeros(size(batch_distance));
            %for b=1:batch_size
                %for c=1:image_num
                    %batch_distance_test(b,c)=sum((batch((a-1)*batch_size+b,:)-image_set(c,:)).^2);
                %end
            %end
            
            [sorted_distance,index]=sort(batch_distance,2);
            for b=bottom_num:top_num
                % 1st order will always be itself
                for order=2:neighbor_num+1
                    dt_matrix(b,index(b-bottom_num+1,order))=sorted_distance(b-bottom_num+1,order);
                    dt_matrix(index(b-bottom_num+1,order),b)= sorted_distance(b-bottom_num+1,order); 
                end
            end
        end
    end
    
elseif strcmp(mode,'adjacency_ball')
    % adjacency ball mode, which uses norm function
    disp('Using adjacency ball to build Laplacian eigenmap');
    ball_radius=params.ball_radius;
    norm=params.norm;
    batch_size=params.batch_size;
    image_num=params.image_num;
    vector_num=params.vector_num;
    dt_matrix=zeros(image_num,image_num);
    
    % batch processing to prevent memory depletion
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
            batch_distance=real(sqrt(repmat(temp1.',[1 size(temp2,2)])+repmat(temp2,[size(temp1,2) 1]) - 2*temp3));
            [sorted_distance,index]=sort(batch_distance,2);
            for b=bottom_num:top_num
                % 1st rank will always be itself
                order=2;
                while (sorted_distance(b-bottom_num+1,order)<ball_radius)&&(order<size(image_set,2))
                    dt_matrix(b,index(b-bottom_num+1,order))=sorted_distance(b-bottom_num+1,order);
                    dt_matrix(index(b-bottom_num+1,order),b)= sorted_distance(b-bottom_num+1,order); 
                    order=order+1;
                end
            end
        end
    end
end

S=zeros(size(dt_matrix));
[dt_row,dt_col,dt_value]=find(dt_matrix);
distance_power=params.distance_power;
reduced_dim=params.reduced_dim;
image_num=params.image_num;
precision=params.precision;
if distance_power>0
    for i=1:length(dt_row)  
        S(dt_row(i),dt_col(i))=exp(-dt_value(i)^2/distance_power);
        % set neighboring elements by inverse expontential function
    end
elseif distance_power==0
    for i=1:size(dt_row)  
        S(dt_row(i),dt_col(i))=1;     
        % set neighboring elements to 1 will reduce computational cost
        % precision will be greatly reduced
    end
end
temp_D=sum(S,2);   
D=diag(temp_D);
L=D-S;

% MATLAB has a bad precision in matrix multiplication
% Here A=image_set.'*L*image_set and B=image_set.'*D*image_set are both asymmetric
%A=image_set.'*L*image_set;
%B=image_set.'*D*image_set;
% Use basic calcualtion to compute A, B for higher precision
% Or consider the average of A+A.' as A, and B+B.' as B if low precision is acceptable
A=zeros(reduced_dim);
B=zeros(reduced_dim);
if strcmp(precision,'high')
    A_temp=zeros(reduced_dim,image_num);    
    B_temp=zeros(reduced_dim,image_num);
    image_set_T=image_set.';
    for a=1:reduced_dim
        for b=1:image_num
            A_temp(a,b)=image_set_T(a,:)*L(:,b);
            B_temp(a,b)=image_set_T(a,:)*D(:,b);
        end
    end
    for a=1:reduced_dim
        for b=1:reduced_dim
            A(a,b)=A_temp(a,:)*image_set(:,b);
            B(a,b)=B_temp(a,:)*image_set(:,b);
        end
    end
elseif strcmp(precision,'low')
    A=image_set.'*L*image_set;
    A=(A+A.')/2;
    B=image_set.'*D*image_set;
    B=(B+B.')/2;
end 


% Several solutions for eigen decomposition
% eig is slow and not satisfying in accuracy
%[L_eigenvector,L_eigenvalue]=eig(A,B,'chol','vector');

% options for eigen-decomposition
opts.maxit=1000;
opts.spdB=1;
opts.tol=1e-15;
[L_eigenvector,L_eigenvalue]=eigs(A,vector_num,'sm',opts);