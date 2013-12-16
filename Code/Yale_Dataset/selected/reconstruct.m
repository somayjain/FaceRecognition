function reconstruct(test_image)
    
    imgs = dir(['.' '/' '*.pgm']);
    p = 192; q = 168; k = 25;
    
    %test_image = 'yaleB01_P00A-005E-10.pgm';
    %test_image = 'Images/face1.jpg'; 
    
    four_fold(imgs, test_image, 0, p, q, k)
    
end

function output_image = four_fold(imgs, test_image, fold_no, p, q, k)

    NumImgs = size(imgs,1);

    flag = 0;
    index = 0;
    labels = cell(NumImgs,1);

    for i=1:NumImgs
        if(mod(i,4) ~= fold_no)
            image = double(imread(imgs(i).name));

            % do a resize here.
            image = imresize(image, [p, q]);

            % Convert the given image to column vector.
            b = transpose(image);
            b = reshape(b, [], 1);

            index = index + 1;
            A(:,index) = b;
            labels{index} = imgs(i).name(1:7);

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
    
    if(size(image, 3) ~=1)
        image = rgb2gray(image);
    end
    

    % do a resize here.
    image = imresize(image, [p, q]);
    image2 = image;

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
    recons = transpose(recons);
    subplot(1, 2, 1)
    imshow(uint8(image2));
    subplot(1, 2, 2)
    imshow(recons);
    
    out_filename = './Images/reconstruction_with_%s.jpg';
    out_filename= sprintf(out_filename,test_image(size(test_image, 2)-8:size(test_image, 2)-4));
    
    print('-djpeg90', out_filename)
    
    output_image = recons;
    
    
end

%img = reshape(mean, 168, 192);
%img = transpose(img); 
%img = uint8(img);