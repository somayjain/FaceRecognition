function PCA_student()
    
    imgs = dir(['./train' '/' '*.bmp']);
    
    
    p = 80; q = 80;
    idx = 0;
    for i=5:5:60
        idx = idx + 1;
        k(idx) = i;
        a1 = hold_one(imgs, 0, p, q, i)
        accuracy(idx) = a1;
    end
    figure();
    plot(k, accuracy, '--b*');
    xlabel('Number of eigenvectors', 'FontSize', 12)
    ylabel('Hold One Out Accuracy', 'FontSize', 12)
    print('-djpeg90', 'Images/Graph.jpg')
    
    
    
end

function accuracy = hold_one(imgs, fold_no, p, q, k)

    NumImgs = size(imgs,1);

    flag = 0;
    index = 0;
    labels = cell(NumImgs,1);
    map_obj = containers.Map;
    
    folder_name = './train/%s';

    for i=1:NumImgs
        
        filename = sprintf(folder_name, imgs(i).name);

        image = double(imread(filename));
        
        % do a resize here.
        image = imresize(image, [p, q]);

        % Convert the given image to column vector.
        b = transpose(image);
        b = reshape(b, [], 1);

        index = index + 1;
        A(:,index) = b;
        labels{index} = imgs(i).name(1:9);
        mylabels(index) = str2num(imgs(i).name(1:9));
        map_obj(imgs(i).name(1:9)) = int32((index-1)/15);

        if(flag==0)
            sum = b;
            flag = 1;
        else
            sum = sum + b;
        end

    end
    
%     imgs_modified = dir(['./modified' '/' '*.bmp']);
%     folder_name = './modified/%s';
%     
%     NumImgs = size(imgs_modified,1);
%     
%     for i=1:NumImgs
%         
%         filename = sprintf(folder_name, imgs_modified(i).name);
% 
%         image = double(imread(filename));
%         
%         % do a resize here.
%         image = imresize(image, [p, q]);
% 
%         % Convert the given image to column vector.
%         b = transpose(image);
%         b = reshape(b, [], 1);
% 
%         index = index + 1;
%         A(:,index) = b;
%         labels{index} = imgs_modified(i).name(1:9);
%         mylabels(index) = str2num(imgs_modified(i).name(1:9));
%         map_obj(imgs_modified(i).name(1:9)) = int32((index-1)/15);
% 
%         if(flag==0)
%             sum = b;
%             flag = 1;
%         else
%             sum = sum + b;
%         end
% 
%     end
    
    
    % Find the mean image.
    mean = sum /index;
    
    % Plot the mean image.
    out_filename = './Images/mean_image_without_rotation.jpg';
    mean_img = reshape(mean, q, p);
    mean_img = transpose(mean_img); 
    mean_img = uint8(mean_img);
    figure();
    imshow(mean_img);
    print('-djpeg90', out_filename)

    % Subtract the mean image from all images.
    for i=1:index
        A(:,i) = A(:, i) - mean;
    end

    At = transpose(A);
    X = At * A;

    % Compute the eigen vector, eigen values of the AtA
    [V,D] = eig(X);
    [D order] = sort(diag(D),'descend'); 
    V = V(:,order);

    % select top k eigen vectors.
    E = V(:, 4:k+4);

    % Compute the eigen faces.
    E = A * E;
    
    % Plot the eigen faces
    rows = k /5;
    out_filename = './Images/eigenfaces_without_rotation_k_%d.jpg';
    out_filename= sprintf(out_filename,k);
    figure();
    for i=1:k
        eigen_face = E(:, i);
        minn = min(eigen_face);
        maxx = max(eigen_face);
        eigen_face = eigen_face - minn;
        eigen_face = eigen_face/ (maxx - minn);
        eigen_face = eigen_face * 255;
        eigen_face = uint8(eigen_face);
        eigen_face = reshape(eigen_face, q, p);
        eigen_face = transpose(eigen_face);
        Faces(:, :, i) = eigen_face;
        subplot(rows, 5, i);
        imshow(eigen_face);    
        
    end
    print('-djpeg90', out_filename)
    

    % Normalise the eigen faces.
    for i=1:k
        E(:, i) = E(:, i)/norm(E(:, i));
    end
    
    % Compute the Feature vector for each class.
    for i=1:index
        for l= 1:k
            FV(i, l) = transpose(E(:, l)) * A(:, i);
        end
    end
    
    % Write to file
        S = strcat('train_data',num2str(fold_no));
        f1 = fopen(S,'w');  
        f3 = fopen('data','w');
        for i=1:index
            fprintf(f1,'%d ',mylabels(i));
            fprintf(f3,'%d ',mylabels(i));
            for l= 1:k
                fprintf(f1,'%d:%f ',l,FV(i,l));
                fprintf(f3,'%d:%f ',l,FV(i,l));
            end
            fprintf(f1,'\n');
            fprintf(f3,'\n');
        end
        %fclose(f1);
    
    %%%%%%%%%% Training Done %%%%%%%%%%%%%%%
    
    S1 = strcat('test_data',num2str(fold_no));
    f2 = fopen(S1,'w');
    
    correct = 0;
    total = 0;
    
    if(fold_no ==0)
        fold_no =4;
    end
    
    imgs2 = dir(['./test' '/' '*.bmp']);
    folder_name = './test/%s';
    NumImgs = size(imgs2,1);
    
    for i=1:NumImgs

        filename = sprintf(folder_name, imgs2(i).name);
        image = double(imread(filename));

        % do a resize here.
        image = imresize(image, [p, q]);

        % Convert the image to column matrix/
        b = transpose(image);
        b = reshape(b, [], 1);

        % Subtract the mean image.
        b = b - mean;

        % Comput the Feature vector for the testing image.
        testFV = zeros(1, k);
        for j=1:k
            testFV(j) = transpose(E(:, j)) * b;
        end
        
        % Writing to file for SVM.
        fprintf(f2,'%d ',str2num(imgs(i).name(1:9)));
        fprintf(f3,'%d ',str2num(imgs(i).name(1:9)));
        for l= 1:k
            fprintf(f2,'%d:%f ',l,testFV(l));
            fprintf(f3,'%d:%f ',l,testFV(l));
        end
        fprintf(f2,'\n');
        fprintf(f3,'\n');

        %fclose(f2);
        %fclose('all');
        %fclose(f3);
        
        % Reconstruct the image from eigen faces
        recons = zeros(size(E(:,1)));
        for j=1:k
            recons = recons + testFV(j) * E(:, j);
        end
        recons = uint8(recons + mean);
        recons = reshape(recons, q, p);
        recons = transpose(recons);
        %imshow(recons);

        % Compute k-NN on the testing image.
        mindx = 1;
        minn = 10000;

        for j=1:index
            if(j==1)
                minn = norm((testFV - FV(j, :)));
            else
                temp = norm((testFV - FV(j, :)));
                if(temp  < minn)
                    mindx = j;
                    minn = temp;
                end
            end

        end
        
        [imgs2(i).name(1:9) imgs(mindx).name(1:9)]
        
        if(strcmpi(imgs2(i).name(1:9), imgs(mindx).name(1:9)) ==1)
            correct = correct +1;
        end

        total = total + 1;
    end
    fclose('all');
    accuracy = double(correct/total) * 100;
    
end

%img = reshape(mean, 168, 192);
%img = transpose(img); 
%img = uint8(img);