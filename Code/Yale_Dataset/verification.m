
    function verification(test_image, claim_label)
        %clear;
        imgs = dir(['./selected' '/' '*.pgm']);
        p = 192; q = 168; k = 25;
        verify(imgs, 0, p, q, k, test_image, claim_label)
    end
     
    function [] = verify(imgs, fold_no, p, q, k, test_image, claim_label)
     
        NumImgs = size(imgs,1);
     
        
        flag = 0;
        index = 0;
        labels = cell(NumImgs,1);
        map_obj = containers.Map;

        folder_name = './selected/%s';
     
        for i=1:NumImgs
            if(mod(i,4) ~= fold_no)
                filename = sprintf(folder_name, imgs(i).name);

                image = double(imread(filename));
     
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
        end
       
        % Find the mean image.
        mean = sum /index;
       
        % Plot the mean image.
        mean_img = reshape(mean, 168, 192);
        mean_img = transpose(mean_img);
        mean_img = uint8(mean_img);
        %imshow(mean_img);
     
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
       
        % Have to plot eigen faces.
       
    %     E_plot = E(:, 1) .* 255;
    %     E_plot
    %     E_plot = uint8(E_plot);
    %    
    %     img = reshape(E_plot, 168, 192);
    %     img = transpose(img);
    %    
    %     imshow(img);
       
     
        % Compute the Feature vector for each class.
        for i=1:index
            for l= 1:k
                FV(i, l) = transpose(E(:, l)) * A(:, i);
            end
        end
       
       
        %%%%%%%%%% Training Done %%%%%%%%%%%%%%%
       
        correct = 0;
        total = 0;
        cnt = 1;
       
        if(fold_no ==0)
            fold_no = 4;
        end
       
        for i=fold_no:4:NumImgs
     
            filename = sprintf(folder_name, imgs(i).name);

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
            
            TFV(cnt,:) = testFV;
            test_label(cnt) = str2num(imgs(i).name(6:7));
            cnt = cnt + 1;
        end
               
     
        % Finding ROC curve
        
        cnt = 0;
        a_cnt = 0; b_cnt = 0;
        while ( b_cnt < size(TFV,1) )
            b_cnt = b_cnt + 1;
            cnt = cnt + 1;
            a_cnt = a_cnt + 1;
            ROC_labels(cnt) = 1;
            dist = 0.0;
            assert(mylabels(a_cnt)==test_label(b_cnt));
            for z = 1:k
                dist = dist + ((TFV(b_cnt,z)-FV(a_cnt,z))*(TFV(b_cnt,z)-FV(a_cnt,z)));
            end
            ROC_dist(cnt) = norm(TFV(b_cnt)-FV(a_cnt));
            cnt = cnt + 1;
            a_cnt = a_cnt + 1;
            ROC_labels(cnt) = 1;
            dist = 0.0;
            assert(mylabels(a_cnt)==test_label(b_cnt));
            for z = 1:k
                dist = dist + ((TFV(b_cnt,z)-FV(a_cnt,z))*(TFV(b_cnt,z)-FV(a_cnt,z)));
            end
            ROC_dist(cnt) = norm(TFV(b_cnt)-FV(a_cnt));
            %ROC_dist(cnt) = sqrt(dist);
            cnt = cnt + 1;
            a_cnt = a_cnt + 1;
            ROC_labels(cnt) = 1;
            dist = 0.0;
            assert(mylabels(a_cnt)==test_label(b_cnt));
            for z = 1:k
                dist = dist + ((TFV(b_cnt,z)-FV(a_cnt,z))*(TFV(b_cnt,z)-FV(a_cnt,z)));
            end
            ROC_dist(cnt) = norm(TFV(b_cnt)-FV(a_cnt));
            %ROC_dist(cnt) = sqrt(dist);
        end
        a_cnt = 20; b_cnt = 0;
        train_sz = size(FV,1);
        while ( b_cnt < size(TFV,1) )
            b_cnt = b_cnt + 1;
            cnt = cnt + 1;
            a_cnt = mod(a_cnt,train_sz)+1;
            ROC_labels(cnt) = 0;
            dist = 0.0;
            assert(mylabels(a_cnt)~=test_label(b_cnt));
            for z = 1:k
                dist = dist + ((TFV(b_cnt,z)-FV(a_cnt,z))*(TFV(b_cnt,z)-FV(a_cnt,z)));
            end
            ROC_dist(cnt) = norm(TFV(b_cnt)-FV(a_cnt));
            %ROC_dist(cnt) = sqrt(dist);
            cnt = cnt + 1;
            a_cnt = mod(a_cnt,train_sz)+1;
            ROC_labels(cnt) = 0;
            dist = 0.0;
            assert(mylabels(a_cnt)~=test_label(b_cnt));
            for z = 1:k
                dist = dist + ((TFV(b_cnt,z)-FV(a_cnt,z))*(TFV(b_cnt,z)-FV(a_cnt,z)));
            end
            %ROC_dist(cnt) = sqrt(dist);
            ROC_dist(cnt) = norm(TFV(b_cnt)-FV(a_cnt));
            cnt = cnt + 1;
            a_cnt = mod(a_cnt,train_sz)+1;
            ROC_labels(cnt) = 0;
            dist = 0.0;
            assert(mylabels(a_cnt)~=test_label(b_cnt));
            for z = 1:k
                dist = dist + ((TFV(b_cnt,z)-FV(a_cnt,z))*(TFV(b_cnt,z)-FV(a_cnt,z)));
            end
            %ROC_dist(cnt) = sqrt(dist);
            ROC_dist(cnt) = norm(TFV(b_cnt)-FV(a_cnt));
        end
        %ROC_labels
       
        [ROC_x, ROC_y, ROC_t, ROC_k, ROC_opt] = perfcurve(ROC_labels, ROC_dist, 0); 
       
        plot(ROC_x, ROC_y);
        title('ROC Curve for CMU-PIE');
        xlabel('False Positive rate');
        ylabel('True Positive rate');
        print('-djpeg90', 'Images/ROC_curve.jpg')
       
       
        for z = 1:size(ROC_x,1)
            if (ROC_x(z,1) == ROC_opt(1,1) && ROC_y(z,1) == ROC_opt(1,2))
                req_thresh = ROC_t(z,1);
            end
        end
        req_thresh
        
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
        
        if(minn > req_thresh)
            disp('No');        
        else
            if(strcmpi(claim_label, labels(mindx)) ==1)
                disp('Yes');
            else
                disp('No');
            end
        end
        
    end
     
    %img = reshape(mean, 168, 192);
    %img = transpose(img);
    %img = uint8(img);

