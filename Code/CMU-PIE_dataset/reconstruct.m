function reconstruct(test_image)
    
    load CMUPIEData;
    
    p = 32; q = 32; k = 25;
    
    output_image = four_fold(CMUPIEData, test_image, 0, p, q, k)
end

function output_image = four_fold(CMUPIEData, test_image, fold_no, p, q, k)

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
    image2 = image;

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

    % Reconstruct the image from eigen faces
    recons = zeros(size(E(:,1)));
    for j=1:k
        recons = recons + testFV(j) * E(:, j);
    end
    recons = uint8(recons + mean);
    recons = reshape(recons, q, p);
    
    
    subplot(1, 2, 1)
    imshow(uint8(image2));
    subplot(1, 2, 2)
    imshow(recons);
    
    out_filename = './Images/reconstructed_image.jpg';
    
    print('-djpeg90', out_filename)
    
    output_image = recons;


    
end

%mean_image = reshape(mean, 168, 192);
%mean_image = transpose(mean_image); 