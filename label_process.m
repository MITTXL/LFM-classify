test_index = [10 29 30 17 25];
a = [1 2 3 4 5];
for k = 0:7
    sst_test(5*k+a,:,:) = sst(test_index+k*50, :, :);
    label_test(5*k+a, :) = label(test_index+k*50, :);
    sst(test_index+k*50, :, :) = [];
    label(test_index+k*50, :) = [];
end