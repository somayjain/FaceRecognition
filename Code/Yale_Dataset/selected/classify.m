function label = classify(test_image)
    
    imgs = dir(['.' '/' '*.pgm']);
    p = 192; q = 168; k = 25;
   
    label = find_label(imgs, test_image, p, q, k);
    
end

function out_label = find_label(imgs, test_image, p, q, k)

    NumImgs = size(imgs,1);

    flag = 0;
    index = 0;
    labels = cell(NumImgs,1);
    map_obj = containers.Map;

    for i=1:NumImgs
        
        image = double(imread(imgs(i).name));

        % do a resize here.
        image = imresize(image, [p, q]);

        % Convert the given image to column vector.
        b = transpose(image);
        b = reshape(b, [], 1);

        index = index + 1;
        A(:,index) = b;
        labels{index} = imgs(i).name(1:7);
        mylabels(index) = str2num(imgs(i).name(6:7));
        map_obj(imgs(i).name(1:7)) = int32((index-1)/15);

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

%img = reshape(mean, 168, 192);
%img = transpose(img); 
%img = uint8(img);