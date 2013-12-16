function label = classify(test_image)
    
    load CMUPIEData;
    
    p = 32; q = 32; k = 25;
    
    label = find_label(CMUPIEData, test_image, p, q, k)
    
end

function out_label = find_label(CMUPIEData, test_image, p, q, k)

    NumImgs = size(CMUPIEData,2);

    flag = 0;
    index = 0;
    fold_no = 0;
    for i=1:NumImgs
        
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
    
    % Find the mean image.
    mean = sum /index;
    
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
    
    
    
    image = double(imread(test_image));

    % do a resize here.
    image = imresize(image, [p, q]);
    
    image = reshape(image, [1 p*q]);

    % Convert the image to column matrix/
    b = transpose(image);
    
    % Subtract the mean image.
    b = b - mean;

    % Comput the Feature vector for the testing image.
    testFV = zeros(1, k);
    for j=1:k
        testFV(j) = transpose(E(:, j)) * b;
    end

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
    
    out_label = labels(mindx);
    
end

%mean_image = reshape(mean, 168, 192);
%mean_image = transpose(mean_image); 