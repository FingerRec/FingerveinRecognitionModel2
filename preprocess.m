%预处理，旋转，灰度化，resize大小为64x64
function  out = preprocess(img)

[m,n] = size(img);
if m > n
    img = imrotate(img,-90);
end
%
if(ndims(img)==3)
    grayimg = rgb2gray(img);
else
    grayimg = img;
end

%grayimg = im2double(grayimg);
out = imresize(grayimg,[64 64]);
%imshow(out)
