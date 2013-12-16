function PCA()
    clear;
    load CMUPIEData;
    
    p = 32; q = 32;
    
    idx = 0;
    
    for i=5:5:45
        idx = idx + 1;
        k(idx) = i;
        a1 = four_fold(CMUPIEData, 0, p, q, i)
        a2 = four_fold(CMUPIEData, 1, p, q, i)
        a3 = four_fold(CMUPIEData, 2, p, q, i)
        a4 = four_fold(CMUPIEData, 3, p, q, i)
    
        mean_accuracy = (a1 + a2 + a3 + a4)/4
        accuracy(idx) = mean_accuracy;
    end
    figure();
    plot(k, accuracy, '--b*');
    xlabel('Number of eigenvectors', 'FontSize', 12)
    ylabel('Four Fold Accuracy', 'FontSize', 12)
    print('-djpeg90', 'Images/Graph.jpg')
    
end

function accuracy = four_fold(CMUPIEData, fold_no, p, q, k)

    NumImgs = size(CMUPIEData,2);

    flag = 0;
    index = 0;

    for i=1:NumImgs
        if(mod(i,4) ~= fold_no)
            image = double(CMUPIEData(i).pixels);

            % do a resize here.
            % image = imresize(image, [p, q]);

            % Convert the given image to column vector.
            b = transpose(image);
            index = index + 1;
            
            A(:,index) = b;
            labels(index) = CMUPIEData(i).label;

            if(flag==0)
                sum = b;
                flag = 1;
            else
                sum = sum + b;
            end
        end
    end
    
    % Find the mean image.
    mean = sum /index;
    
    % Plot the mean image.
    out_filename = './Images/mean_image_with_fold_no_%d.jpg';
    out_filename= sprintf(out_filename,fold_no);
    mean_img = reshape(mean, q, p);
%     mean_img = transpose(mean_img); 
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
    out_filename = './Images/eigenfaces_with_fold_no_%d_k_%d.jpg';
    out_filename= sprintf(out_filename,fold_no,k);
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
%         eigen_face = transpose(eigen_face);
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

    %%%%%%%%%% Training Done %%%%%%%%%%%%%%%
    
    correct = 0;
    total = 0;
    
    if(fold_no ==0)
        fold_no =4;
    end
    
    for i=fold_no:4:NumImgs

        image = double(CMUPIEData(i).pixels);

        % do a resize here.
        %image = imresize(image, [p, q]);

        % Convert the image to column matrix/
        b = transpose(image);
        %b = reshape(b, [], 1);

        % Subtract the mean image.
        b = b - mean;

        % Comput the Feature vector for the testing image.
        testFV = zeros(1, k);
        for j=1:k
            testFV(j) = transpose(E(:, j)) * b;
        end
        
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

        if(CMUPIEData(i).label==labels(mindx))
            correct = correct +1;
        end

        total = total + 1;
    end
    accuracy = double(correct/total) * 100;
    
end

%mean_image = reshape(mean, 168, 192);
%mean_image = transpose(mean_image); 